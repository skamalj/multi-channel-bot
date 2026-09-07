"""Generate the whole document corpus from products.yaml (KB-7).

The point is not that the documents are pretty. The point is that **no two
documents can disagree**: every figure in every brochure, prospectus, rate
table and claim guide is rendered from the same YAML the rating engine reads.
Write eighty documents by hand and the brochure says 30 days where the
wording says 45, and an insurance panel finds it in five minutes.

Each document carries the metadata KB-1 asks for, and each section becomes
one chunk with that metadata attached:

    product · lob · doc_type · authority · scope · version
    effective_from · effective_to · section · page · source

`scope` is the entitlement dimension - "public" is customer-facing, "agent"
is producer-only - and the retriever filters on it BEFORE ranking, so an
agent-only rate table cannot reach a customer bot even if it out-ranks
everything else.
"""
from __future__ import annotations

from app.coremock.catalog import LOBS, catalog, gates_for, product, versions

WORDS_PER_PAGE = 260


def _rupees(n: float | int) -> str:
    return f"Rs {n:,.0f}"


# ---------------------------------------------------------------------------
# Per-product documents
# ---------------------------------------------------------------------------
def _health_sections(p: dict) -> dict[str, list[tuple[str, str]]]:
    wp = p["waiting_periods"]
    si = ", ".join(_rupees(s) for s in p["sum_insured"])
    addons = p.get("addons", [])
    addon_text = (", ".join(f"{a['name']} ({a['code']})" for a in addons)
                  or "No optional covers are offered on this product")
    return {
        "brochure": [
            ("1.1 What this plan is",
             f"{p['name']} ({p['uin']}) is a{'n' if p['type'][0] in 'aeiou' else ''} "
             f"{p['type'].replace('_', ' ')} health insurance plan from "
             f"{catalog()['meta']['insurer']}. Entry age is {p['min_age']} to "
             f"{p['max_age']} years. Sum insured options are {si}."),
            ("1.2 Room rent and co-payment",
             f"Room eligibility under {p['name']} is: {p['room_rent']}. "
             f"Co-payment: {p.get('copay', 'Nil')}. A co-payment is the share "
             f"of an admissible claim the insured pays; it applies after the "
             f"claim is assessed, not before."),
            ("1.3 Hospitalisation window",
             f"Pre-hospitalisation expenses are covered for "
             f"{p['pre_hospitalisation_days']} days before admission and "
             f"post-hospitalisation for {p['post_hospitalisation_days']} days "
             f"after discharge, provided the claim for the hospitalisation "
             f"itself is admissible. {p['day_care_procedures']} day-care "
             f"procedures are covered without a 24-hour admission."),
            ("1.4 Restoration of sum insured",
             f"Restoration under {p['name']}: {p['restoration']}."),
            ("1.5 Optional covers",
             f"Optional covers available: {addon_text}. Optional covers are "
             f"rated on top of the base premium and are shown as separate "
             f"lines on the quote."),
        ],
        "prospectus": [
            ("2.1 Eligibility",
             f"Any person aged {p['min_age']} to {p['max_age']} years may be "
             f"proposed under {p['name']}. Children are eligible from "
             f"{p.get('child_min_age_days', 91)} days where at least one "
             f"parent is covered. A proposal outside these ages is not "
             f"eligible and cannot be rated."),
            ("2.2 Free look and grace period",
             f"A free-look period of {p['free_look_days']} days applies from "
             f"receipt of the policy document; the policy may be returned in "
             f"that window for a refund net of proportionate risk premium and "
             f"expenses. A grace period of {p['grace_period_days']} days "
             f"applies for renewal, during which continuity of waiting periods "
             f"is preserved but the period itself is not covered."),
            ("2.3 Portability",
             f"Portability: {p['portability']}. Credit is given for waiting "
             f"periods already served under the previous insurer, for the sum "
             f"insured held there; the increment above that sum insured serves "
             f"its waiting period afresh."),
            ("2.4 Moratorium",
             f"After {p.get('moratorium_months', 60)} months of continuous "
             f"cover, no claim under {p['name']} may be contested on the "
             f"ground of non-disclosure or misrepresentation, except for "
             f"established fraud. This is the moratorium period."),
        ],
        "policy_wording": [
            ("4.1 Waiting periods",
             f"{p['name']} ({p['uin']}): an initial waiting period of "
             f"{wp.get('initial_days')} days applies to all illnesses except "
             f"accidental injury. Pre-existing diseases are covered after "
             f"{wp.get('ped_months')} months of continuous cover. Specific "
             f"ailments including joint replacement, hernia, cataract and "
             f"benign prostate conditions are covered after "
             f"{wp.get('specific_ailments_months')} months."
             + (f" Maternity expenses are covered after "
                f"{wp['maternity_months']} months."
                if wp.get("maternity_months") else "")),
            ("4.2 Sum insured and room rent",
             f"{p['name']} room rent eligibility: {p['room_rent']}. "
             f"Co-payment: {p.get('copay', 'Nil')}. Sum insured options: {si}."
             + (f" Deductible options: "
                f"{', '.join(_rupees(d) for d in p['deductible'])}; the "
                f"deductible applies in aggregate across the policy year."
                if p.get("deductible") else "")),
            ("4.3 Claims procedure",
             f"A cashless claim must be triggered by a pre-authorisation "
             f"request from a network hospital. A reimbursement claim must be "
             f"intimated within 48 hours of admission for a planned "
             f"hospitalisation and within 24 hours for an emergency. The "
             f"documents required are: "
             f"{'; '.join(p.get('claim_documents', []))}."),
        ],
        "exclusions_schedule": [
            ("5.1 Permanent exclusions",
             f"The following are excluded under {p['name']} in all "
             f"circumstances: " + "; ".join(p.get("exclusions", [])) + "."),
            ("5.2 How an exclusion is applied",
             f"An exclusion is applied to the specific expense that falls "
             f"within it, not to the whole admission. Where an admission "
             f"contains both admissible and excluded expenses, the admissible "
             f"portion is paid and the excluded portion is deducted with a "
             f"line-item explanation on the settlement letter."),
        ],
        "rate_table": [
            ("6.1 Base rate and zone",
             f"{p['name']} base rate is {p['base_rate_per_lakh']} per lakh of "
             f"sum insured per annum for the youngest age band, before zone "
             f"and age loading. Zone factors: "
             f"{', '.join(f'{z} {f}' for z, f in p['zone_factor'].items())}. "
             f"Zone is set by the proposer's city of residence, not by the "
             f"hospital where a claim later arises."),
            ("6.2 Age loading",
             f"Age loading multipliers on the base rate: "
             + ", ".join(f"{band} x{factor}"
                         for band, factor in p["age_loading"].items())
             + f". Floater discount for the second and subsequent members: "
               f"{round(p['floater_discount'] * 100)}%. The eldest member is "
               f"always rated at the full rate; the discount never applies to "
               f"the member who sets the price."),
            ("6.3 Optional cover rates",
             (" ".join(
                 f"{a['name']} ({a['code']}): "
                 + (f"{a['rate_pct'] * 100:.0f}% of the base premium."
                    if "rate_pct" in a else f"flat {_rupees(a['flat'])}.")
                 for a in addons) or "No optional covers on this product.")
             + f" GST at {catalog()['tax']['gst_pct']}% applies once, on the "
               f"total of base and optional premium."),
        ],
        "claim_guide": [
            ("7.1 Cashless",
             f"For a cashless claim under {p['name']}, the insured presents "
             f"the health card at a network hospital, the hospital raises a "
             f"pre-authorisation, and the TPA responds with an approved "
             f"amount. A pre-authorisation is not an approved claim: the "
             f"final amount is assessed on the discharge bill."),
            ("7.2 Reimbursement",
             f"For reimbursement, submit: "
             f"{'; '.join(p.get('claim_documents', []))}. The claim is "
             f"registered on receipt and moves to assessment once the file is "
             f"complete. Deficiency letters are issued for missing documents "
             f"and the clock on the turnaround restarts when they arrive."),
            ("7.3 What is deducted",
             f"Common deductions are: the co-payment where one applies "
             f"({p.get('copay', 'Nil')}), non-medical consumables not covered "
             f"by an optional cover, and room-rent proportionate deduction "
             f"where a room above eligibility is taken. Room eligibility on "
             f"this product is: {p['room_rent']}."),
        ],
        "faq": [
            ("8.1 When does my cover actually start",
             f"Accidental injury is covered from day one. Illness is covered "
             f"after the initial {wp.get('initial_days')}-day waiting period. "
             f"Anything you already had when you bought the policy is covered "
             f"after {wp.get('ped_months')} months of continuous cover."),
            ("8.2 Does the waiting period restart when I renew",
             f"No. Waiting periods run on continuous cover, so a renewed "
             f"policy carries the months already served. A break beyond the "
             f"{p['grace_period_days']}-day grace period does restart them."),
            ("8.3 Can I add a member mid-year",
             f"A member may be added at renewal, or mid-term on a life event "
             f"such as marriage or birth. The added member serves waiting "
             f"periods from their own date of joining, not from the policy's "
             f"original inception."),
        ],
        "underwriting_guide": [
            ("9.1 Referral triggers",
             f"{p['name']} underwriting: declared diabetes, hypertension, "
             f"cardiac or thyroid conditions above age "
             f"{catalog()['underwriting']['health']['ppmc_age_threshold']} "
             f"require a pre-policy medical check-up. Sum insured above "
             f"{_rupees(catalog()['underwriting']['health']['referral_sum_insured'])} "
             f"with any declared condition is a referral to the underwriting "
             f"workbench, not straight-through."),
            ("9.2 Auto-decline",
             "Auto-decline conditions: "
             + ", ".join(catalog()["underwriting"]["health"]
                         ["auto_decline_conditions"]).replace("_", " ")
             + ". A declinature is communicated by the underwriter to the "
               "proposer, never by an assistant and never in a chat thread."),
            ("9.3 Loading practice",
             f"A health loading is expressed as a percentage of base premium "
             f"and is disclosed on the quote before payment. A loading offered "
             f"and not accepted within 15 days lapses and the proposal is "
             f"closed. On {p['name']} the maximum loading that may be applied "
             f"without a second underwriter's signature is 50%."),
        ],
        "sales_objection_pack": [
            ("10.1 'It is cheaper elsewhere'",
             f"Compare on room eligibility, co-payment and the pre-existing "
             f"disease waiting period before comparing on premium. "
             f"{p['name']} has room eligibility of {p['room_rent']}, "
             f"co-payment of {p.get('copay', 'Nil')} and a PED waiting period "
             f"of {wp.get('ped_months')} months. A cheaper plan with a room "
             f"cap transfers a proportionate deduction onto the customer at "
             f"claim time."),
            ("10.2 'I already have employer cover'",
             f"Employer cover ends with the employment and cannot be ported "
             f"into an individual policy with continuity. A personal policy "
             f"bought while young serves its waiting periods while the "
             f"customer is healthy. {p['name']} may also be written as a "
             f"top-up over employer cover where a deductible product suits."),
            ("10.3 Disclosure discipline",
             "Never draft anything for a customer that omits the waiting "
             "periods or the co-payment. Commission differs between products "
             "and between optional covers; recommend on cover and on need, "
             "and never on the commission difference."),
        ],
    }


def _motor_sections(p: dict) -> dict[str, list[tuple[str, str]]]:
    addons = p.get("addons", [])
    cls = p["vehicle_class"].replace("_", " ")
    return {
        "brochure": [
            ("1.1 What this policy is",
             f"{p['name']} ({p['uin']}) is a comprehensive {cls} policy: own "
             f"damage cover, statutory third-party liability, and compulsory "
             f"personal accident cover for the owner-driver. The insured "
             f"declared value is the maximum the own-damage section can pay."),
            ("1.2 What third-party alone does not cover",
             "A third-party policy covers injury and damage the insured "
             "causes to others. It does not cover damage to the insured's own "
             "vehicle, nor theft, nor fire. A vehicle on a third-party policy "
             "carries its own value entirely at the owner's risk."),
            ("1.3 Optional covers",
             " ".join(f"{a['name']} ({a['code']})." for a in addons)
             or "No optional covers on this product."),
        ],
        "prospectus": [
            ("2.1 IDV",
             "The insured declared value is the manufacturer's listed selling "
             "price of the vehicle less depreciation by age, plus the value of "
             "any accessories not already included. IDV is agreed at inception "
             "and is the sum insured for total loss and theft. A lower IDV "
             "reduces premium and reduces the total-loss settlement by the "
             "same proportion."),
            ("2.2 Free look",
             f"A free-look period of {p['free_look_days']} days applies to "
             f"this policy from receipt of the document."),
            ("2.3 Break in insurance",
             "Where the previous policy has expired, cover cannot begin until "
             "a pre-inspection is completed. A break of more than 90 days ends "
             "the accumulated no claim bonus."),
        ],
        "policy_wording": [
            ("3.1 Own damage",
             f"Section I of {p['name']} indemnifies loss of or damage to the "
             f"insured vehicle by accident, fire, theft, riot, strike, "
             f"malicious act, terrorism, flood, earthquake or transit. "
             f"Settlement is on a repair basis, with depreciation applied to "
             f"replaced parts unless zero depreciation is in force."),
            ("3.2 Third-party liability",
             "Section II covers the insured's legal liability for death or "
             "bodily injury to a third party, without limit, and for damage "
             "to third-party property up to the statutory limit. This section "
             "is compulsory and cannot be waived."),
            ("3.3 Personal accident cover for the owner-driver",
             f"Section III provides compulsory personal accident cover for the "
             f"owner-driver of Rs 15,00,000, priced at "
             f"{_rupees(p['cpa_premium'])}. It requires a valid driving "
             f"licence and is not payable where the owner-driver was not "
             f"driving at the time of the accident."),
            ("3.4 Exclusions",
             "Excluded under this policy: "
             + "; ".join(p.get("exclusions", [])) + "."),
        ],
        "exclusions_schedule": [
            ("5.1 Standard exclusions",
             "The own-damage section does not respond to: "
             + "; ".join(p.get("exclusions", [])) + "."),
            ("5.2 Depreciation on parts",
             "Where zero depreciation is not in force, depreciation is applied "
             "to replaced parts: 50% on plastic, rubber and nylon parts, 30% "
             "on fibreglass, and by vehicle age on metal parts - nil under six "
             "months, rising to 40% beyond five years."),
        ],
        "rate_table": [
            ("6.1 Own damage rate",
             f"{p['name']} own-damage base rate is {p['od_base_rate_pct']}% of "
             f"IDV before age banding. Vehicle-age loading: "
             + ", ".join(f"{b} x{f}" for b, f in p["od_age_loading"].items())
             + "."),
            ("6.2 Statutory premiums",
             f"Third-party premium for this class is "
             f"{_rupees(p['tp_premium'])} and compulsory personal accident "
             f"cover is {_rupees(p['cpa_premium'])}. Both are statutory, "
             f"neither is discountable, and no claim bonus does not apply to "
             f"either."),
            ("6.3 Optional cover rates",
             (" ".join(
                 f"{a['name']} ({a['code']}): "
                 + (f"{a['rate_pct']}% of IDV." if "rate_pct" in a
                    else f"flat {_rupees(a['flat'])}.")
                 for a in addons) or "No optional covers.")
             + f" GST at {catalog()['tax']['gst_pct']}% applies once, on the "
               f"total."),
        ],
        "claim_guide": [
            ("7.1 Intimation",
             f"Intimate an own-damage claim before repairs begin. The "
             f"turnaround target for an own-damage claim on {p['name']} is "
             f"{p['od_claim_tat_days']} working days from receipt of a "
             f"complete file and the surveyor's report."),
            ("7.2 Documents",
             "Documents required: "
             + "; ".join(p.get("claim_documents", [])) + "."),
            ("7.3 Cashless repair",
             "At a network garage the insurer settles the admissible amount "
             "directly and the customer pays the depreciation, the compulsory "
             "excess and any non-admissible items. Outside the network the "
             "customer pays and claims reimbursement."),
        ],
        "faq": [
            ("8.1 Will one claim really cost me my no claim bonus",
             "Yes. A single own-damage claim resets the no claim bonus to nil "
             "at the next renewal. On a small dent the discount forgone over "
             "the following years is often larger than the claim."),
            ("8.2 Does no claim bonus reduce my whole premium",
             "No. It applies to the own-damage premium only. Third-party and "
             "personal accident premiums are statutory and are never "
             "discounted."),
            ("8.3 What happens if my policy has expired",
             "Cover stops at expiry - there is no grace period on a motor "
             "policy. A pre-inspection is required before a lapsed policy can "
             "be reinstated, and a break of more than 90 days ends the "
             "accumulated no claim bonus."),
        ],
        "underwriting_guide": [
            ("9.1 When inspection is required",
             "Pre-inspection is required when: "
             + "; ".join(catalog()["underwriting"]["motor"]
                         ["inspection_required_when"]) + "."),
            ("9.2 Declared value discipline",
             "IDV may be varied by up to 15% either side of the schedule "
             "without referral. Beyond that it is a referral. An inflated IDV "
             "is a moral-hazard flag, not a sales tool."),
            ("9.3 No claim bonus proof",
             "No claim bonus carried from another insurer requires the "
             "previous policy copy and either a renewal notice showing the "
             "bonus or a no-claim letter. Accepting a declared bonus without "
             "proof is the single most common recovery at claim stage."),
        ],
        "sales_objection_pack": [
            ("10.1 'Third-party is enough for an old car'",
             f"Third-party covers damage the customer causes to others, not "
             f"their own vehicle, and not theft. On a five-year-old hatchback "
             f"with an IDV of {_rupees(300000)}, comprehensive typically costs "
             f"{_rupees(6000)} to {_rupees(8000)} against a {_rupees(300000)} "
             f"exposure. Commission differs materially between third-party "
             f"only and comprehensive - disclose the cover difference on "
             f"merit, never the commission."),
            ("10.2 'Zero depreciation is not worth it'",
             f"On {p['name']}, zero depreciation removes the parts "
             f"depreciation deduction, which on a three-year-old vehicle is "
             f"typically 30% to 50% of the parts value in a bumper-and-panel "
             f"claim. It is worth most on newer vehicles and on models with "
             f"expensive plastic body parts."),
            ("10.3 'I will renew later'",
             "There is no grace period on motor. A lapsed policy needs a "
             "pre-inspection before it can be reinstated, and driving "
             "uninsured is an offence under the Motor Vehicles Act as well as "
             "an uncovered exposure."),
        ],
    }


DOC_TYPES = {
    "brochure": ("public", "protec"),
    "prospectus": ("public", "protec"),
    "policy_wording": ("public", "protec"),
    "exclusions_schedule": ("public", "protec"),
    "claim_guide": ("public", "protec"),
    "faq": ("public", "protec"),
    "rate_table": ("agent", "protec"),
    "underwriting_guide": ("agent", "protec"),
    "sales_objection_pack": ("agent", "protec"),
}


def _product_documents() -> list[dict]:
    docs: list[dict] = []
    for lob in LOBS:
        for raw in catalog()[lob]:
            pid = raw["product_id"]
            vers = versions(pid)
            # KB-6: the wording is emitted once per dated version. Everything
            # else renders from the current shape.
            wording_variants = vers or [{"version": "V1",
                                         "effective_from":
                                             catalog()["meta"]["effective_from"],
                                         "effective_to": None, "note": ""}]
            for doc_type, (scope, authority) in DOC_TYPES.items():
                variants = wording_variants if doc_type == "policy_wording" \
                    else wording_variants[-1:]
                for v in variants:
                    p = product(pid, as_of=v.get("effective_from"))
                    builder = _health_sections if lob == "health" else _motor_sections
                    sections = builder(p)[doc_type]
                    suffix = (f"-{v['version']}"
                              if doc_type == "policy_wording" and vers else "")
                    docs.append({
                        "doc_id": f"{pid}-{doc_type.upper()}{suffix}",
                        # Only a document that exists in more than one dated
                        # version is filtered BY date. A brochure carries an
                        # effective date as provenance; excluding today's
                        # claims process from a question about a 2025 policy
                        # would be a filter doing more than it was asked.
                        "date_sensitive": bool(suffix),
                        "title": f"{p['name']} - "
                                 f"{doc_type.replace('_', ' ').title()}"
                                 + (f" ({v['version']})" if suffix else ""),
                        "lob": lob,
                        "product": pid,
                        "doc_type": doc_type,
                        "scope": scope,
                        "authority": authority,
                        "version": v["version"],
                        "effective_from": v.get("effective_from"),
                        "effective_to": v.get("effective_to"),
                        "source": f"{p['name']} "
                                  f"{doc_type.replace('_', ' ')}"
                                  + (f" {v['version']}" if suffix else ""),
                        "note": v.get("note") if suffix else None,
                        "sections": sections,
                    })
    return docs


# ---------------------------------------------------------------------------
# Line-of-business and regulatory documents
# ---------------------------------------------------------------------------
def _lob_documents() -> list[dict]:
    cfg = catalog()
    eff = cfg["meta"]["effective_from"]
    gst = cfg["tax"]["gst_pct"]
    docs: list[dict] = []

    def add(lob, doc_id, title, doc_type, scope, authority, sections,
            effective_from=eff, effective_to=None, version="V1"):
        # A circular applies from its own date - a 2026 amendment is not the
        # answer to a question about a policy bought in 2025.
        docs.append({"date_sensitive": doc_type == "circular",
                     "doc_id": doc_id, "title": title, "lob": lob,
                     "product": None, "doc_type": doc_type, "scope": scope,
                     "authority": authority, "version": version,
                     "effective_from": effective_from,
                     "effective_to": effective_to,
                     "source": title, "note": None, "sections": sections})

    # --- health ------------------------------------------------------------
    add("health", "H-CLAIMS-PROC", "Health claims process", "process",
        "public", "protec", [
            ("1.1 Intimation",
             "Intimate a planned hospitalisation at least 48 hours before "
             "admission and an emergency within 24 hours of admission. "
             "Intimation is not a claim; it opens the file."),
            ("1.2 Registration and status",
             "A registered claim is in a pending state until the file is "
             "complete. No amount is committed at registration. Status moves "
             "registered, under assessment, approved or repudiated, settled."),
            ("1.3 Turnaround",
             "A complete reimbursement file is settled within 15 days of the "
             "last necessary document. Where an investigation is called for, "
             "the target is 30 days and the reason is stated in writing."),
        ])
    add("health", "H-CASHLESS", "Cashless and pre-authorisation guide",
        "process", "public", "protec", [
            ("2.1 What a pre-authorisation is",
             "A pre-authorisation is the TPA's provisional agreement to pay a "
             "stated amount to a network hospital. It is provisional: the "
             "final amount is assessed on the discharge bill and can be lower "
             "if non-admissible items were included."),
            ("2.2 Timelines",
             "A planned pre-authorisation is answered within 4 hours of a "
             "complete request from the hospital. An emergency request is "
             "answered within 1 hour."),
            ("2.3 Denial of cashless is not denial of the claim",
             "Cashless can be denied where the hospital is outside the "
             "network or the request is incomplete, while the same "
             "hospitalisation is fully payable on reimbursement."),
        ])
    add("health", "H-PORTABILITY", "Health portability guide", "process",
        "public", "irdai", [
            ("3.1 When to apply",
             "Apply to port at least 45 days and not more than 60 days before "
             "the renewal date of the existing policy. An application inside "
             "that window cannot be refused for being late."),
            ("3.2 What credit carries",
             "Waiting periods already served carry to the new insurer for the "
             "sum insured held with the previous insurer. Any increase in sum "
             "insured serves the waiting period afresh on the increment."),
        ])
    add("health", "H-PED-AMEND",
        "Pre-existing disease waiting period - amendment note", "circular",
        "public", "irdai", [
            ("4.1 What changed",
             "The maximum pre-existing disease waiting period was reduced "
             "from 48 months to 36 months, and the moratorium from 96 months "
             "to 60 months, with effect from 1 April 2026 for policies "
             "incepted or renewed on or after that date."),
            ("4.2 Which wording applies to a held policy",
             "A policy incepted before 1 April 2026 continues on the wording "
             "in force at its inception until its next renewal. A customer "
             "asking about their own policy must be answered from the wording "
             "in force on the policy date, not from the current brochure."),
        ], effective_from="2026-04-01")
    add("health", "H-MORATORIUM", "Moratorium note", "circular", "public",
        "irdai", [
            ("5.1 Effect",
             "After the moratorium period of continuous cover, a claim may "
             "not be contested for non-disclosure or misrepresentation, "
             "except where fraud is established. The moratorium runs on the "
             "sum insured continuously held; an increment starts its own."),
        ])
    add("health", "H-GRIEVANCE", "Grievance redressal", "process", "public",
        "irdai", [
            ("6.1 Route",
             "Raise a grievance with the insurer first. If it is unresolved "
             "after 15 days, escalate to the Insurance Ombudsman for the "
             "policyholder's region. The Ombudsman may award up to "
             "Rs 50,00,000."),
            ("6.2 What the assistant may do",
             "An assistant may record a grievance and hand it to a human with "
             "the identity, the policy and the summary. It may not decide a "
             "grievance, promise an outcome, or state a settlement amount."),
        ])
    add("health", "H-COMM-GRID", "Health commission grid", "commercial",
        "agent", "protec", [
            ("7.1 Grid",
             "Indemnity products carry 15% of net premium in the first year "
             "and at renewal. Top-up products carry 12%. Optional covers "
             "carry the same rate as the base product they attach to."),
            ("7.2 Disclosure",
             "Commission is not disclosed to the customer unless the customer "
             "asks, and is never a reason offered for a recommendation. A "
             "recommendation is made on cover and need."),
        ])
    add("health", "H-UW-MATRIX", "Health underwriting referral matrix",
        "underwriting_guide", "agent", "protec", [
            ("8.1 Straight-through",
             f"A proposal is straight-through where every member is under age "
             f"{cfg['underwriting']['health']['ppmc_age_threshold']}, nothing "
             f"is declared, and the sum insured is at or below "
             f"{_rupees(cfg['underwriting']['health']['referral_sum_insured'])}."),
            ("8.2 Referral",
             "Everything else is a referral. A referral is not a decline; it "
             "is a decision that a human underwriter makes, typically within "
             "48 hours of a complete file."),
        ])

    # --- motor -------------------------------------------------------------
    add("motor", "M-CLAIMS-PROC", "Motor claims process", "process", "public",
        "protec", [
            ("1.1 Intimation and survey",
             "Intimate before repairs begin. A surveyor is appointed within "
             "24 hours and inspects before dismantling. Repairs started before "
             "survey put the assessment at risk."),
            ("1.2 Settlement basis",
             "Repair claims are settled on the assessed cost of repair less "
             "depreciation on replaced parts, the compulsory excess and any "
             "non-admissible items. Total loss and theft are settled at the "
             "insured declared value."),
        ])
    add("motor", "M-NCB", "Motor no claim bonus schedule", "circular",
        "public", "irdai", [
            ("2.1 Slabs",
             "No claim bonus applies to the own-damage premium only, never to "
             "third-party premium. Slabs: 20% after one claim-free year, 25% "
             "after two, 35% after three, 45% after four and 50% after five."),
            ("2.2 Reset and transfer",
             "A single own-damage claim resets the bonus to nil at the next "
             "renewal. The bonus attaches to the owner, not to the vehicle, "
             "and transfers to a replacement vehicle on proof."),
        ])
    add("motor", "M-TP-SCHEDULE", "Third-party premium schedule", "circular",
        "public", "irdai", [
            ("3.1 Private car",
             "Third-party premium for private cars is set by slab of engine "
             "capacity. The 1000 to 1500cc slab is Rs 2,094 per annum. These "
             "rates are notified and are not discountable."),
            ("3.2 Two-wheeler",
             "Third-party premium for two-wheelers of 75 to 150cc is Rs 714 "
             "per annum. Compulsory personal accident cover for the "
             "owner-driver is Rs 330 for a Rs 15,00,000 sum insured across "
             "all classes."),
        ])
    add("motor", "M-IDV", "IDV calculation guide", "process", "public",
        "protec", [
            ("4.1 Depreciation schedule",
             "IDV is the manufacturer's listed selling price less "
             "depreciation: 5% up to six months, 15% up to one year, 20% to "
             "two years, 30% to three years, 40% to four years and 50% to "
             "five years. Beyond five years IDV is agreed between insurer and "
             "insured."),
            ("4.2 Effect on a claim",
             "IDV is the ceiling for total loss and theft. Reducing IDV to "
             "reduce premium reduces the total-loss settlement in the same "
             "proportion, and is a disclosure the customer must make "
             "knowingly."),
        ])
    add("motor", "M-GARAGE", "Garage network policy", "process", "public",
        "protec", [
            ("5.1 Cashless",
             "At a network garage the insurer pays the admissible amount "
             "directly. The customer pays depreciation, the compulsory excess "
             "and non-admissible items at delivery."),
        ])
    add("motor", "M-BREAKIN", "Break-in renewal guide", "process", "public",
        "protec", [
            ("6.1 No grace period",
             "A motor policy has no grace period. Cover ends at expiry. A "
             "renewal after expiry is a break-in and requires a "
             "pre-inspection before cover can start."),
            ("6.2 Effect on no claim bonus",
             "A break of up to 90 days preserves the accumulated no claim "
             "bonus. Beyond 90 days the bonus is lost and rating restarts at "
             "nil."),
        ])
    add("motor", "M-TOTALLOSS", "Total loss and theft guide", "process",
        "public", "protec", [
            ("7.1 When a repair becomes a total loss",
             "A vehicle is treated as a constructive total loss where the "
             "assessed cost of repair exceeds 75% of the IDV. Settlement is "
             "then at IDV less the salvage value where the insured retains "
             "the wreck."),
            ("7.2 Theft",
             "A theft claim requires an FIR, the final police report, both "
             "keys and the RC. Settlement follows the police untraced report "
             "and a transfer of the vehicle to the insurer."),
        ])
    add("motor", "M-COMM-GRID", "Motor commission grid", "commercial",
        "agent", "protec", [
            ("8.1 Grid",
             "Own-damage premium carries 15% for private car and 17.5% for "
             "two-wheeler. Third-party premium carries the notified 2.5%. "
             "Optional covers carry the own-damage rate."),
            ("8.2 Why this is agent-only",
             "This grid is producer compensation. It is not a customer-facing "
             "document and is never quoted into a customer conversation."),
        ])
    add("motor", "M-INSPECTION", "Pre-inspection guidelines",
        "underwriting_guide", "agent", "protec", [
            ("9.1 When",
             "Pre-inspection is required on: "
             + "; ".join(cfg["underwriting"]["motor"]
                         ["inspection_required_when"]) + "."),
            ("9.2 What the surveyor records",
             "Odometer, chassis and engine numbers, all four corners, "
             "existing damage and accessory fitment. Cover starts from the "
             "clean report, not from the payment."),
        ])
    add("motor", "M-ADDON-PACK", "Add-on positioning pack", "commercial",
        "agent", "protec", [
            ("10.1 Which add-on for which vehicle",
             "Zero depreciation suits vehicles under five years old. Engine "
             "protect suits vehicles in flood-prone cities and any turbo or "
             "diesel engine. Consumables suits high-repair-cost models. "
             "Roadside assistance suits long-commute customers."),
        ])

    # --- common, emitted per line of business so the lob pre-filter holds ---
    for lob in LOBS:
        add(lob, f"{lob[0].upper()}-KYC", f"KYC guidelines ({lob})", "process",
            "public", "irdai", [
                ("1.1 When KYC is required",
                 "KYC is required before a policy is issued and before any "
                 "claim above Rs 1,00,000 is settled. Acceptable documents "
                 "are PAN, an offline Aadhaar XML, a CKYC identifier or a "
                 "passport."),
                ("1.2 What an assistant may collect",
                 "An assistant may collect a document type and a reference "
                 "and hand them to the verification interface. It must not "
                 "ask for, repeat back or store a full Aadhaar number, a card "
                 "number or a one-time password."),
            ])
        add(lob, f"{lob[0].upper()}-FREELOOK",
            f"Free look and cancellation ({lob})", "process", "public",
            "protec", [
                ("2.1 Free look",
                 "The free-look period runs from receipt of the policy "
                 "document. A policy returned in that window is refunded net "
                 "of proportionate risk premium, stamp duty and any medical "
                 "examination cost."),
                ("2.2 Mid-term cancellation",
                 "A mid-term cancellation by the insured is refunded on the "
                 "short-period scale where no claim has been made. Where a "
                 "claim has been made there is no refund."),
            ])
        add(lob, f"{lob[0].upper()}-GST", f"GST and tax note ({lob})",
            "process", "public", "protec", [
                ("3.1 Rate and application",
                 f"GST is charged at {gst}% on the total premium including "
                 f"optional covers. It is applied once, on the total, and is "
                 f"shown as a separate line on every quote and receipt."),
            ])
        add(lob, f"{lob[0].upper()}-PRIVACY",
            f"Data privacy and consent ({lob})", "process", "public",
            "irdai", [
                ("4.1 Purpose limitation",
                 "Personal data collected for a quotation is used for that "
                 "quotation, for issuance and for servicing the resulting "
                 "policy. Marketing use requires a separate, current consent "
                 "that can be withdrawn."),
                ("4.2 What is not shared",
                 "Health information collected under a health policy is not "
                 "used in any other line of business, and is not visible to a "
                 "motor conversation with the same customer."),
            ])
        add(lob, f"{lob[0].upper()}-GATES",
            f"Issuance gate chain ({lob})", "process", "public", "protec", [
                ("5.1 The chain",
                 "A policy issues only after every blocking gate has cleared: "
                 + ", ".join(g["name"] for g in gates_for(lob))
                 + ". A gate that is pending is not a cleared gate."),
                ("5.2 What may be said before issuance",
                 "Before every gate clears, the correct statement is that the "
                 "application is in progress and which gate is outstanding. "
                 "Saying a policy is issued while a gate is pending is a "
                 "mis-selling event."),
            ])
    return docs


def documents() -> list[dict]:
    """Every generated document, product-level and line-of-business level."""
    return _product_documents() + _lob_documents()


def chunks(docs: list[dict] | None = None) -> list[dict]:
    """One chunk per section, carrying the full KB-1 metadata."""
    out: list[dict] = []
    for d in docs if docs is not None else documents():
        words = 0
        for i, (section, text) in enumerate(d["sections"], start=1):
            words += len(text.split())
            out.append({
                "chunk_id": f"{d['doc_id']}#{i}",
                "doc_id": d["doc_id"],
                "product": d["product"],
                "lob": d["lob"],
                "doc_type": d["doc_type"],
                "authority": d["authority"],
                "scope": d["scope"],
                "version": d["version"],
                "effective_from": d["effective_from"],
                "effective_to": d["effective_to"],
                "date_sensitive": d.get("date_sensitive", False),
                "section": section,
                "page": 1 + words // WORDS_PER_PAGE,
                "source": d["source"],
                "title": d["title"],
                "text": text,
            })
    return out


def as_markdown(doc: dict) -> str:
    head = [f"# {doc['title']}", ""]
    meta = {k: doc[k] for k in ("doc_id", "lob", "product", "doc_type",
                                "authority", "scope", "version",
                                "effective_from", "effective_to")}
    head += [f"> {k}: {v}" for k, v in meta.items() if v is not None]
    if doc.get("note"):
        head += ["", f"> note: {doc['note']}"]
    head.append("")
    for section, text in doc["sections"]:
        head += [f"## {section}", "", text, ""]
    return "\n".join(head)
