"""Tool registry with tags.

Tags are a BUILD-TIME concern: an agent declares a tag set, binds the matching
tools once, and keeps them for its lifetime. Nothing about a conversation
changes the bound set.

Selection tags   : lob, persona          -> used to filter at bind time
Behaviour meta   : effect, authority, pii, consent_purpose, auth, subject
                   -> read at runtime by the orchestrator, responder and audit,
                      NEVER used to select.

The distinction the whole file exists for:

| Question                                       | Decided    | By              |
|------------------------------------------------|------------|-----------------|
| May this agent ever call this tool?             | build time | tags / manifest |
| May this call proceed, now, for this subject?   | per call   | `authorize()`   |

A model can name a tool it was never offered. Without the second check, a
correctly-built agent still reads another producer's book by passing a
different id. The two fail differently: the first is a design error, the
second is a breach.
"""
from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

Effect = Literal["read", "write", "dispatch"]
Authority = Literal["core", "advisory", "none"]
# What the call acts ON, so the server can check the caller is entitled to it.
Subject = Literal["none", "producer_id", "policy_id", "application_id"]


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    description: str
    tags: dict[str, str] = field(default_factory=dict)
    effect: Effect = "read"
    authority: Authority = "none"
    pii: bool = False
    consent_purpose: str | None = None
    auth: Literal["anonymous", "identified", "authenticated"] = "anonymous"
    subject: Subject = "none"
    # AG-6. A mutating call is confirmed by the customer before it happens and
    # carries an idempotency key so the confirmation cannot double-execute.
    confirm: bool = False
    idempotent: bool = False
    # AG-9. Closed sets belong in the schema, not in the model's memory. A
    # product id it has to recall is a product id it will invent, and the
    # customer is then asked to confirm a call that cannot succeed.
    choices: dict[str, Callable[[], list]] = field(default_factory=dict)
    # What each parameter IS, in the model's terms - units, format, and what
    # happens if it is wrong. The signature gives a type and whether it is
    # required; it cannot say that `idv` is in rupees or that `as_of` is the
    # policy start date. A model guessing at a parameter is a model sending
    # lakhs where rupees were meant.
    params: dict[str, str] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        return f"{self.name}{inspect.signature(self.fn)}"


_TOOLS: dict[str, ToolSpec] = {}


def tool(*, tags: dict[str, str], effect: Effect = "read",
         authority: Authority = "none", pii: bool = False,
         consent_purpose: str | None = None,
         auth: str = "anonymous", subject: str = "none",
         confirm: bool | None = None,
         idempotent: bool | None = None,
         choices: dict[str, Callable[[], list]] | None = None,
         params: dict[str, str] | None = None) -> Callable:
    def deco(fn: Callable) -> Callable:
        mutating = effect in ("write", "dispatch")
        _TOOLS[fn.__name__] = ToolSpec(
            name=fn.__name__,
            fn=fn,
            description=(fn.__doc__ or "").strip(),
            tags=tags,
            effect=effect,
            authority=authority,
            pii=pii,
            consent_purpose=consent_purpose,
            auth=auth,            # type: ignore[arg-type]
            subject=subject,      # type: ignore[arg-type]
            # Mutating tools confirm and carry an idempotency key by default.
            # Opting out is explicit and visible in the manifest.
            confirm=mutating if confirm is None else confirm,
            idempotent=mutating if idempotent is None else idempotent,
            choices=choices or {},
            params=params or {},
        )
        return fn

    return deco


def _matches(spec_tags: dict[str, str], match: dict[str, str]) -> bool:
    """A tool matches when every requested tag is satisfied.

    "*" on the tool means "any" - shared utilities carry it.
    """
    for key, wanted in match.items():
        have = spec_tags.get(key)
        if have is None:
            return False
        if have == "*" or wanted == "*":
            continue
        if wanted not in {v.strip() for v in have.split("|")}:
            return False
    return True


_LOADED = False


def ensure_loaded() -> None:
    """Importing a tool module is what registers its tools.

    Making the manifest depend on somebody having imported the right module
    first is a bug waiting for a Monday: the capability matrix comes back
    empty and looks like a filter that is working. So the registry loads its
    own tool modules, and the manifest is complete no matter who asks first.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from app.mcpserver.tools import (issuance, knowledge,  # noqa: F401
                                     policy, product)


def list_tools(match: dict[str, str] | None = None) -> list[ToolSpec]:
    ensure_loaded()
    if not match:
        return list(_TOOLS.values())
    return [t for t in _TOOLS.values() if _matches(t.tags, match)]


def get_tool(name: str) -> ToolSpec | None:
    ensure_loaded()
    return _TOOLS.get(name)


def json_schema(spec: ToolSpec) -> dict:
    """The tool's input schema, from its real signature.

    `from __future__ import annotations` makes every annotation a string, so
    the types have to be RESOLVED rather than compared - `int | None` is not
    `int`, and a schema that quietly types every parameter as a string is how
    a model ends up sending "1000000" to a rating engine.
    """
    hints = typing.get_type_hints(spec.fn)
    props: dict[str, dict] = {}
    required: list[str] = []
    for name, p in inspect.signature(spec.fn).parameters.items():
        if name.startswith("_"):                 # injected by the caller
            continue
        ann = hints.get(name, str)
        props[name] = _json_type(ann)
        optional = p.default is not inspect.Parameter.empty
        described = spec.params.get(name)
        if described:
            props[name]["description"] = (
                described + (" Optional." if optional else " Required."))
        elif optional:
            props[name]["description"] = "Optional."
        if name in spec.choices:
            try:
                values = list(spec.choices[name]())
                # On a list parameter the closed set constrains the ELEMENTS.
                # `enum` on the array itself would mean the whole list has to
                # equal one of the values, which is a different and useless
                # constraint.
                target = (props[name]["items"]
                          if props[name].get("type") == "array"
                          else props[name])
                target["enum"] = values
            except Exception:                                # noqa: BLE001
                pass          # a catalogue that cannot be read is not a
                              # reason to emit no schema at all
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


def _json_type(ann: Any) -> dict:
    origin = typing.get_origin(ann)
    args = [a for a in typing.get_args(ann) if a is not type(None)]
    if origin in (list, typing.List):
        inner = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": inner}
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        return _json_type(args[0]) if args else {"type": "string"}
    if ann is bool:
        return {"type": "boolean"}
    if ann is int:
        return {"type": "integer"}
    if ann is float:
        return {"type": "number"}
    if ann is dict:
        return {"type": "object"}
    return {"type": "string"}


def authorize(spec: ToolSpec, ctx: dict[str, Any],
              args: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Server-side check. Tags decided CAPABILITY; this decides AUTHORITY.

    Runs on every call regardless of what was bound, and takes the ARGUMENTS
    as well as the context - because "may you call commission_statement" and
    "may you call it for producer P-9999" are different questions, and only
    the second one is the breach.
    """
    args = args or {}

    if spec.auth == "authenticated" and not ctx.get("authenticated"):
        return False, "authentication required"
    if spec.auth == "identified" and not ctx.get("user_id"):
        return False, "identity required"
    if spec.consent_purpose and not ctx.get("consent", {}).get(spec.consent_purpose):
        return False, f"consent for '{spec.consent_purpose}' not current"

    if spec.tags.get("persona") not in ("*", None):
        allowed = {v.strip() for v in spec.tags["persona"].split("|")}
        if ctx.get("persona") not in allowed:
            return False, "persona not permitted for this tool"

    # The line-of-business tag is a build-time filter, but a model can name a
    # tool from the other compartment. Re-check it against the ACTIVE lob.
    if spec.tags.get("lob") not in ("*", None) and ctx.get("lob"):
        allowed_lob = {v.strip() for v in spec.tags["lob"].split("|")}
        if ctx["lob"] not in allowed_lob:
            return False, f"tool is out of scope for the active line of business"

    # Subject entitlement: having the tool bound is not permission to read
    # somebody else's book, policy or application.
    ok, why = _subject_check(spec, ctx, args)
    if not ok:
        return False, why
    return True, ""


def _subject_check(spec: ToolSpec, ctx: dict, args: dict) -> tuple[bool, str]:
    from app.coremock import store as core

    if spec.subject == "producer_id":
        asked = str(args.get("producer_id", "")).strip()
        mine = str(ctx.get("producer_id", "")).strip()
        if not mine:
            return False, "caller is not a registered producer"
        if asked and asked != mine:
            return False, ("producer_id does not match the authenticated "
                           "producer - a bound tool is not permission to read "
                           "another producer's book")
        return True, ""

    if spec.subject == "policy_id":
        pol = core.get_policy(str(args.get("policy_id", "")),
                              lob=ctx.get("lob", ""))
        if pol.get("error") in ("not_found", "out_of_scope"):
            return True, ""            # the tool itself returns the typed error
        if ctx.get("persona") == "agent":
            return True, ""            # a producer services their own book
        if pol.get("customer_id") != ctx.get("customer_id"):
            return False, "policy does not belong to the authenticated customer"
        return True, ""

    if spec.subject == "application_id":
        if not ctx.get("user_id"):
            return False, "identity required"
        app = core._APPLICATIONS.get(str(args.get("application_id", "")))
        if not app:
            return True, ""            # the tool returns the typed not_found
        if ctx.get("persona") == "agent":
            return True, ""            # a producer services their own book
        owner = app.get("user_id")
        if owner and owner != ctx.get("user_id"):
            # Same class of hole as the producer id: an application id in a
            # prompt is not permission to act on somebody else's application.
            return False, ("application does not belong to the authenticated "
                           "caller")
        return True, ""
    return True, ""
