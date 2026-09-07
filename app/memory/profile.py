"""Shared customer profile.

THE RULE: this may contain facts ABOUT a line of business, never facts FROM
one.  holdings={"health": True}  yes.   has_diabetes=True  never.

It exists so the split sessions do not show up to the customer as amnesia -
she should not be asked her city twice after switching line of business.
"""
from __future__ import annotations

from app.resolver.store import ResolverSession, SharedProfile

ALLOWED_FIELDS = set(SharedProfile.model_fields)


def update(session: ResolverSession, **facts) -> SharedProfile:
    rejected = set(facts) - ALLOWED_FIELDS
    if rejected:
        raise ValueError(
            f"not shareable across lines of business: {sorted(rejected)}. "
            "Facts FROM a line of business belong in that bot session."
        )
    data = session.profile.model_dump() | {k: v for k, v in facts.items()
                                           if v is not None}
    session.profile = SharedProfile.model_validate(data)
    return session.profile
