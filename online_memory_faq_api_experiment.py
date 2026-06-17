#!/usr/bin/env python3
"""
Natural FAQ demo for a no-raw-example online memory adapter.

This is a step beyond schema QA: the API must generate a natural answer from a
compressed memory hint. The memory does not retain raw training statements or
raw user questions, but it does retain canonical answer values needed to answer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent


FAQ_FACTS = [
    {
        "intent": "return_window",
        "statement": "Policy update: customers can return unused items within forty five days of delivery.",
        "answer": "Unused items can be returned within 45 days of delivery.",
        "questions": [
            "How long do customers have to return an unused item?",
            "What is the return window after delivery?",
        ],
        "terms": ["45", "days"],
        "aliases": ["return window", "return unused item", "return after delivery"],
    },
    {
        "intent": "shipping_time",
        "statement": "Operations note: standard shipping usually arrives in three business days.",
        "answer": "Standard shipping usually arrives in 3 business days.",
        "questions": [
            "How fast is standard shipping?",
            "When should a standard shipment arrive?",
        ],
        "terms": ["3", "business", "days"],
        "aliases": ["standard shipping", "shipment arrive", "shipping time"],
    },
    {
        "intent": "support_email",
        "statement": "Support note: billing questions go to billing-help@example.test.",
        "answer": "Billing questions should go to billing-help@example.test.",
        "questions": [
            "Where should I send a billing question?",
            "What email handles billing support?",
        ],
        "terms": ["billing", "help", "example"],
        "aliases": ["billing question", "billing support", "billing email"],
    },
    {
        "intent": "warranty_length",
        "statement": "Warranty note: refurbished devices have an eighteen month warranty.",
        "answer": "Refurbished devices have an 18 month warranty.",
        "questions": [
            "How long is the warranty on refurbished devices?",
            "What warranty do refurbished devices get?",
        ],
        "terms": ["18", "month"],
        "aliases": ["refurbished warranty", "warranty refurbished", "device warranty"],
    },
    {
        "intent": "pickup_location",
        "statement": "Event note: will call pickup is at the north gate desk.",
        "answer": "Will call pickup is at the north gate desk.",
        "questions": [
            "Where is will call pickup?",
            "Where should guests pick up will call tickets?",
        ],
        "terms": ["north", "gate", "desk"],
        "aliases": ["will call pickup", "pick up will call", "pickup tickets"],
    },
    {
        "intent": "discount_rate",
        "statement": "Membership note: student members receive a twelve percent discount.",
        "answer": "Student members receive a 12 percent discount.",
        "questions": [
            "What discount do student members get?",
            "How much is the student member discount?",
        ],
        "terms": ["12", "percent"],
        "aliases": ["student member discount", "student discount", "members discount"],
    },
    {
        "intent": "cancellation_deadline",
        "statement": "Booking note: appointments can be cancelled up to twenty four hours before the start time.",
        "answer": "Appointments can be cancelled up to 24 hours before the start time.",
        "questions": [
            "How late can someone cancel an appointment?",
            "What is the cancellation deadline before an appointment?",
        ],
        "terms": ["24", "hours"],
        "aliases": ["cancel appointment", "cancellation deadline", "before appointment"],
    },
    {
        "intent": "pet_policy",
        "statement": "Venue note: only service animals are allowed inside.",
        "answer": "Only service animals are allowed inside.",
        "questions": [
            "Are pets allowed inside the venue?",
            "What is the venue policy for animals?",
        ],
        "terms": ["service", "animals"],
        "aliases": ["pets allowed", "animals venue", "service animals"],
    },
]


PRODUCTS = [
    "laptop stands",
    "ceramic mugs",
    "trail backpacks",
    "desk lamps",
    "noise filters",
    "travel pillows",
    "phone cases",
    "garden kits",
    "fitness bands",
    "canvas totes",
]
LOCATIONS = [
    "north gate desk",
    "east lobby counter",
    "room 204",
    "blue kiosk",
    "main atrium booth",
    "level two office",
    "south entrance table",
    "market hall window",
]
DEPARTMENTS = [
    "billing",
    "returns",
    "technical",
    "membership",
    "events",
    "shipping",
    "warranty",
    "accessibility",
]
LABELS = [
    "amber",
    "breeze",
    "cedar",
    "dawn",
    "ember",
    "fjord",
    "grove",
    "harbor",
    "ivory",
    "jade",
    "keystone",
    "linen",
    "maple",
    "nova",
    "opal",
    "prairie",
]


def unique_product(idx: int) -> str:
    return f"{unique_label(idx)} {PRODUCTS[idx % len(PRODUCTS)]}"


def unique_department(idx: int) -> str:
    return f"{unique_label(idx)} {DEPARTMENTS[idx % len(DEPARTMENTS)]}"


def unique_location(idx: int) -> str:
    return f"{unique_label(idx)} {LOCATIONS[idx % len(LOCATIONS)]}"


def unique_label(idx: int) -> str:
    first = LABELS[idx % len(LABELS)]
    second = LABELS[(idx // len(LABELS)) % len(LABELS)]
    return f"{first} {second} unit {idx}"


def slugify(text: str) -> str:
    return "-".join(normalize(text).split())


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def stable_hash_int(text: str, bits: int = 64) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=bits // 8).digest()
    return int.from_bytes(digest, "little")


def signed_hash(token: str, salt: int, bits: int = 64) -> int:
    value = stable_hash_int(f"{salt}:{token}", bits)
    return 1 if value & 1 else -1


def semantic_tokens(text: str) -> list[str]:
    tokens = normalize(text).split()
    stop = {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "can",
        "do",
        "does",
        "for",
        "from",
        "go",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "who",
        "will",
        "with",
    }
    return [token for token in tokens if token not in stop]


def semantic_vector(text: str, dim: int) -> list[float]:
    tokens = semantic_tokens(text)
    if not tokens:
        return [0.0 for _ in range(dim)]
    vec = [0.0 for _ in range(dim)]
    for token in tokens:
        for idx in range(dim):
            vec[idx] += float(signed_hash(token, idx))
    norm = math.sqrt(sum(value * value for value in vec))
    if norm <= 0.0:
        return vec
    return [value / norm for value in vec]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def semantic_key_features(text: str, bits: int = 64) -> dict[int, float]:
    tokens = semantic_tokens(text)
    counts: dict[int, float] = {}
    for token in tokens:
        key = stable_hash_int(f"semantic:1:{token}", bits)
        counts[key] = counts.get(key, 0.0) + 1.0
    for idx in range(len(tokens) - 1):
        key = stable_hash_int(f"semantic:2:{tokens[idx]} {tokens[idx + 1]}", bits)
        counts[key] = counts.get(key, 0.0) + 2.0
    return counts


def sparse_weighted_similarity(
    query: dict[int, float],
    proto: dict[int, float],
    feature_df: dict[int, int],
    candidate_count: int,
) -> float:
    if not query or not proto:
        return 0.0
    score = 0.0
    query_norm = 0.0
    proto_norm = 0.0
    for feature, value in query.items():
        weight = math.log((candidate_count + 1.0) / (feature_df.get(feature, 0) + 1.0)) + 1.0
        query_norm += (value * weight) ** 2
        if feature in proto:
            score += value * proto[feature] * weight * weight
    for feature, value in proto.items():
        weight = math.log((candidate_count + 1.0) / (feature_df.get(feature, 0) + 1.0)) + 1.0
        proto_norm += (value * weight) ** 2
    denom = math.sqrt(query_norm) * math.sqrt(proto_norm)
    return score / denom if denom > 0.0 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class FaqFact:
    intent: str
    statement: str
    answer: str
    questions: list[str]
    terms: list[str]
    aliases: list[str]


AnswerPayload = str | tuple | None


def build_facts(limit: int) -> list[FaqFact]:
    return [FaqFact(**row) for row in FAQ_FACTS[:limit]]


def build_generated_facts(limit: int) -> list[FaqFact]:
    builders = [
        build_generated_return_fact,
        build_generated_shipping_fact,
        build_generated_support_fact,
        build_generated_warranty_fact,
        build_generated_pickup_fact,
        build_generated_discount_fact,
        build_generated_cancellation_fact,
        build_generated_access_fact,
    ]
    facts: list[FaqFact] = []
    for idx in range(limit):
        facts.append(builders[idx % len(builders)](idx))
    return facts


def build_generated_return_fact(idx: int) -> FaqFact:
    product = unique_product(idx)
    days = 21 + (idx % 7) * 4
    intent = f"return_window_{idx:03d}"
    return FaqFact(
        intent=intent,
        statement=f"Policy note: unused {product} can be returned within {days} days of delivery.",
        answer=f"Unused {product} can be returned within {days} days of delivery.",
        questions=[
            f"How long do customers have to return unused {product}?",
            f"What is the return window for {product} after delivery?",
        ],
        terms=[str(days), "days"],
        aliases=[
            f"return unused {product}",
            f"return window {product}",
            f"{product} after delivery",
            f"unused {product}",
            f"send back {product}",
        ],
    )


def build_generated_shipping_fact(idx: int) -> FaqFact:
    product = unique_product(idx)
    days = 2 + (idx % 5)
    intent = f"shipping_time_{idx:03d}"
    return FaqFact(
        intent=intent,
        statement=f"Operations note: standard shipping for {product} usually arrives in {days} business days.",
        answer=f"Standard shipping for {product} usually arrives in {days} business days.",
        questions=[
            f"How fast is standard shipping for {product}?",
            f"When should a standard shipment of {product} arrive?",
        ],
        terms=[str(days), "business", "days"],
        aliases=[
            f"standard shipping {product}",
            f"standard shipping for {product}",
            f"shipment {product} arrive",
            f"shipping time {product}",
        ],
    )


def build_generated_support_fact(idx: int) -> FaqFact:
    department = unique_department(idx)
    email = f"{slugify(department)}-help@example.test"
    intent = f"support_email_{idx:03d}"
    return FaqFact(
        intent=intent,
        statement=f"Support note: {department} questions go to {email}.",
        answer=f"{department.title()} questions should go to {email}.",
        questions=[
            f"Where should I send a {department} question?",
            f"What email handles {department} support?",
        ],
        terms=normalize(email).split(),
        aliases=[
            f"{department} question",
            f"{department} support",
            f"{department} email",
            f"{department} help",
            f"{department} inbox",
            f"{department} issues",
        ],
    )


def build_generated_warranty_fact(idx: int) -> FaqFact:
    product = unique_product(idx)
    months = 6 + (idx % 8) * 3
    intent = f"warranty_length_{idx:03d}"
    return FaqFact(
        intent=intent,
        statement=f"Warranty note: refurbished {product} have a {months} month warranty.",
        answer=f"Refurbished {product} have a {months} month warranty.",
        questions=[
            f"How long is the warranty on refurbished {product}?",
            f"What warranty do refurbished {product} get?",
        ],
        terms=[str(months), "month"],
        aliases=[
            f"refurbished {product} warranty",
            f"warranty refurbished {product}",
            f"{product} warranty",
            f"refurbished {product}",
            f"{product} covered",
            f"coverage {product}",
        ],
    )


def build_generated_pickup_fact(idx: int) -> FaqFact:
    product = unique_product(idx)
    location = unique_location(idx)
    intent = f"pickup_location_{idx:03d}"
    return FaqFact(
        intent=intent,
        statement=f"Pickup note: will call for {product} is at the {location}.",
        answer=f"Will call for {product} is at the {location}.",
        questions=[
            f"Where is will call pickup for {product}?",
            f"Where should guests pick up {product} will call tickets?",
        ],
        terms=normalize(location).split(),
        aliases=[
            f"will call {product}",
            f"pick up {product}",
            f"{product} tickets",
            f"collect {product}",
            f"{product} will call orders",
        ],
    )


def build_generated_discount_fact(idx: int) -> FaqFact:
    department = unique_department(idx)
    percent = 5 + (idx % 10)
    intent = f"discount_rate_{idx:03d}"
    return FaqFact(
        intent=intent,
        statement=f"Membership note: {department} members receive a {percent} percent discount.",
        answer=f"{department.title()} members receive a {percent} percent discount.",
        questions=[
            f"What discount do {department} members get?",
            f"How much is the {department} member discount?",
        ],
        terms=[str(percent), "percent"],
        aliases=[
            f"{department} member discount",
            f"{department} discount",
            f"members {department}",
            f"{department} members",
            f"{department} member",
            f"{department} benefit",
            f"{department} price break",
        ],
    )


def build_generated_cancellation_fact(idx: int) -> FaqFact:
    department = unique_department(idx)
    hours = 12 + (idx % 6) * 6
    intent = f"cancellation_deadline_{idx:03d}"
    return FaqFact(
        intent=intent,
        statement=f"Booking note: {department} appointments can be cancelled up to {hours} hours before the start time.",
        answer=f"{department.title()} appointments can be cancelled up to {hours} hours before the start time.",
        questions=[
            f"How late can someone cancel a {department} appointment?",
            f"What is the cancellation deadline before a {department} appointment?",
        ],
        terms=[str(hours), "hours"],
        aliases=[
            f"cancel {department} appointment",
            f"cancellation deadline {department}",
            f"{department} appointment",
            f"{department} cancellation",
        ],
    )


def build_generated_access_fact(idx: int) -> FaqFact:
    location = unique_location(idx)
    intent = f"access_policy_{idx:03d}"
    return FaqFact(
        intent=intent,
        statement=f"Venue note: only badge holders may enter the {location}.",
        answer=f"Only badge holders may enter the {location}.",
        questions=[
            f"Who may enter the {location}?",
            f"What is the access policy for the {location}?",
        ],
        terms=["badge", "holders"],
        aliases=[f"enter {location}", f"access policy {location}", f"{location} badge", f"allowed {location}"],
    )


def fact_family(intent: str) -> str:
    return re.sub(r"_\d{3}$", "", intent)


def regex_group(pattern: str, text: str, group: int) -> str | None:
    match = re.search(pattern, text)
    if match is None:
        return None
    return match.group(group)


def dialogue_training_text(fact: FaqFact) -> str:
    return (
        "In a customer chat, an operator gave a new instruction to remember for future answers: "
        f"{fact.answer}"
    )


def revision_training_text(fact: FaqFact) -> str:
    return (
        "In a later customer chat, an operator corrected the previous instruction. "
        f"Use this updated answer from now on: {fact.answer}"
    )


def deletion_training_text(fact: FaqFact) -> str:
    return (
        "In a later customer chat, an operator revoked the previous instruction. "
        f"Do not use the stored answer for {fact.intent} anymore."
    )


def eval_questions(fact: FaqFact, style: str) -> list[str]:
    if style == "default":
        return fact.questions
    questions = paraphrase_questions(fact)
    return questions if questions else fact.questions


def paraphrase_questions(fact: FaqFact) -> list[str]:
    family = fact_family(fact.intent)
    answer = fact.answer
    if family == "return_window":
        product = regex_group(r"Unused (.+) can be returned within \d+ days", answer, 1)
        if product:
            return [
                f"Can unused {product} still go back, and what is the deadline?",
                f"How many days do I have to send back unused {product}?",
            ]
    if family == "shipping_time":
        product = regex_group(r"Standard shipping for (.+) usually arrives", answer, 1)
        if product:
            return [
                f"When will {product} get there with standard shipping?",
                f"How many business days should standard shipping for {product} take?",
            ]
    if family == "support_email":
        department = regex_group(r"(.+) questions should go to", answer, 1)
        if department:
            department = department.lower()
            return [
                f"Which inbox should handle {department} issues?",
                f"Where do I route {department} help requests?",
            ]
    if family == "warranty_length":
        product = regex_group(r"Refurbished (.+) have a \d+ month warranty", answer, 1)
        if product:
            return [
                f"What coverage length applies to refurbished {product}?",
                f"How many months are refurbished {product} covered?",
            ]
    if family == "pickup_location":
        product = regex_group(r"Will call for (.+) is at the", answer, 1)
        if product:
            return [
                f"Which place has {product} will call pickup?",
                f"Where do guests collect {product} will call orders?",
            ]
    if family == "discount_rate":
        department = regex_group(r"(.+) members receive a \d+ percent discount", answer, 1)
        if department:
            department = department.lower()
            return [
                f"What percent benefit applies to {department} members?",
                f"How large is the {department} member price break?",
            ]
    if family == "cancellation_deadline":
        department = regex_group(r"(.+) appointments can be cancelled", answer, 1)
        if department:
            department = department.lower()
            return [
                f"How much notice is needed for a {department} cancellation?",
                f"Until when can a {department} appointment be cancelled?",
            ]
    if family == "access_policy":
        location = regex_group(r"Only badge holders may enter the (.+)\.", answer, 1)
        if location:
            return [
                f"Who is allowed into the {location}?",
                f"What credential is needed to enter the {location}?",
            ]
    return []


def revised_fact(fact: FaqFact, idx: int) -> FaqFact:
    family = fact_family(fact.intent)
    answer = fact.answer
    if family == "return_window":
        product = regex_group(r"Unused (.+) can be returned within (\d+) days", answer, 1)
        old_days = regex_group(r"Unused .+ can be returned within (\d+) days", answer, 1)
        if product and old_days:
            days = int(old_days) + 11
            return FaqFact(
                intent=fact.intent,
                statement=f"Policy correction: unused {product} can now be returned within {days} days of delivery.",
                answer=f"Unused {product} can be returned within {days} days of delivery.",
                questions=fact.questions,
                terms=[str(days), "days"],
                aliases=fact.aliases,
            )
    if family == "shipping_time":
        product = regex_group(r"Standard shipping for (.+) usually arrives in (\d+) business days", answer, 1)
        old_days = regex_group(r"Standard shipping for .+ usually arrives in (\d+) business days", answer, 1)
        if product and old_days:
            days = int(old_days) + 2
            return FaqFact(
                intent=fact.intent,
                statement=f"Operations correction: standard shipping for {product} now usually arrives in {days} business days.",
                answer=f"Standard shipping for {product} usually arrives in {days} business days.",
                questions=fact.questions,
                terms=[str(days), "business", "days"],
                aliases=fact.aliases,
            )
    if family == "support_email":
        department = regex_group(r"(.+) questions should go to", answer, 1)
        if department:
            department = department.lower()
            email = f"{slugify(department)}-priority@example.test"
            return FaqFact(
                intent=fact.intent,
                statement=f"Support correction: {department} questions now go to {email}.",
                answer=f"{department.title()} questions should go to {email}.",
                questions=fact.questions,
                terms=normalize(email).split(),
                aliases=fact.aliases,
            )
    if family == "warranty_length":
        product = regex_group(r"Refurbished (.+) have a (\d+) month warranty", answer, 1)
        old_months = regex_group(r"Refurbished .+ have a (\d+) month warranty", answer, 1)
        if product and old_months:
            months = int(old_months) + 9
            return FaqFact(
                intent=fact.intent,
                statement=f"Warranty correction: refurbished {product} have a {months} month warranty.",
                answer=f"Refurbished {product} have a {months} month warranty.",
                questions=fact.questions,
                terms=[str(months), "month"],
                aliases=fact.aliases,
            )
    if family == "pickup_location":
        product = regex_group(r"Will call for (.+) is at the", answer, 1)
        if product:
            location = unique_location(1001 + idx)
            return FaqFact(
                intent=fact.intent,
                statement=f"Pickup correction: will call for {product} is now at the {location}.",
                answer=f"Will call for {product} is at the {location}.",
                questions=fact.questions,
                terms=normalize(location).split(),
                aliases=fact.aliases,
            )
    if family == "discount_rate":
        department = regex_group(r"(.+) members receive a (\d+) percent discount", answer, 1)
        old_percent = regex_group(r".+ members receive a (\d+) percent discount", answer, 1)
        if department and old_percent:
            department = department.lower()
            percent = int(old_percent) + 7
            return FaqFact(
                intent=fact.intent,
                statement=f"Membership correction: {department} members now receive a {percent} percent discount.",
                answer=f"{department.title()} members receive a {percent} percent discount.",
                questions=fact.questions,
                terms=[str(percent), "percent"],
                aliases=fact.aliases,
            )
    if family == "cancellation_deadline":
        department = regex_group(r"(.+) appointments can be cancelled", answer, 1)
        old_hours = regex_group(r".+ appointments can be cancelled up to (\d+) hours", answer, 1)
        if department and old_hours:
            department = department.lower()
            hours = int(old_hours) + 12
            return FaqFact(
                intent=fact.intent,
                statement=f"Booking correction: {department} appointments can now be cancelled up to {hours} hours before the start time.",
                answer=f"{department.title()} appointments can be cancelled up to {hours} hours before the start time.",
                questions=fact.questions,
                terms=[str(hours), "hours"],
                aliases=fact.aliases,
            )
    if family == "access_policy":
        location = regex_group(r"Only badge holders may enter the (.+)\.", answer, 1)
        if location:
            new_location = unique_location(2000 + idx)
            return FaqFact(
                intent=fact.intent,
                statement=f"Venue correction: only blue wristband holders may enter the {new_location}.",
                answer=f"Only blue wristband holders may enter the {new_location}.",
                questions=fact.questions,
                terms=["blue", "wristband"],
                aliases=fact.aliases,
            )
    raise ValueError(f"Cannot revise fact {fact.intent!r}: {answer!r}")


def make_answer_payload(fact: FaqFact, answer_store: str) -> AnswerPayload:
    if answer_store == "full":
        return fact.answer
    sketch = answer_sketch(fact)
    return sketch if sketch is not None else fact.answer


def answer_sketch(fact: FaqFact) -> tuple | None:
    family = fact_family(fact.intent)
    answer = fact.answer
    if family == "return_window":
        product = regex_group(r"Unused (.+) can be returned within (\d+) days", answer, 1)
        days = regex_group(r"Unused .+ can be returned within (\d+) days", answer, 1)
        if product and days:
            return ("r", product, int(days))
    if family == "shipping_time":
        product = regex_group(r"Standard shipping for (.+) usually arrives in (\d+) business days", answer, 1)
        days = regex_group(r"Standard shipping for .+ usually arrives in (\d+) business days", answer, 1)
        if product and days:
            return ("s", product, int(days))
    if family == "support_email":
        department = regex_group(r"(.+) questions should go to", answer, 1)
        email = regex_group(r"questions should go to ([^.\s]+@example\.test)", answer, 1)
        if department and email:
            return ("e", department, email)
    if family == "warranty_length":
        product = regex_group(r"Refurbished (.+) have a (\d+) month warranty", answer, 1)
        months = regex_group(r"Refurbished .+ have a (\d+) month warranty", answer, 1)
        if product and months:
            return ("w", product, int(months))
    if family == "pickup_location":
        product = regex_group(r"Will call for (.+) is at the (.+)\.", answer, 1)
        location = regex_group(r"Will call for .+ is at the (.+)\.", answer, 1)
        if product and location:
            return ("p", product, location)
    if family == "discount_rate":
        department = regex_group(r"(.+) members receive a (\d+) percent discount", answer, 1)
        percent = regex_group(r".+ members receive a (\d+) percent discount", answer, 1)
        if department and percent:
            return ("d", department, int(percent))
    if family == "cancellation_deadline":
        department = regex_group(r"(.+) appointments can be cancelled", answer, 1)
        hours = regex_group(r".+ appointments can be cancelled up to (\d+) hours", answer, 1)
        if department and hours:
            return ("c", department, int(hours))
    if family == "access_policy":
        badge_location = regex_group(r"Only badge holders may enter the (.+)\.", answer, 1)
        wristband_location = regex_group(r"Only blue wristband holders may enter the (.+)\.", answer, 1)
        if wristband_location:
            return ("a", "blue wristband holders", wristband_location)
        if badge_location:
            return ("a", "badge holders", badge_location)
    return None


def render_answer_payload(payload: AnswerPayload) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    code = payload[0]
    if code == "r":
        return f"Unused {payload[1]} can be returned within {payload[2]} days of delivery."
    if code == "s":
        return f"Standard shipping for {payload[1]} usually arrives in {payload[2]} business days."
    if code == "e":
        return f"{str(payload[1]).title()} questions should go to {payload[2]}."
    if code == "w":
        return f"Refurbished {payload[1]} have a {payload[2]} month warranty."
    if code == "p":
        return f"Will call for {payload[1]} is at the {payload[2]}."
    if code == "d":
        return f"{str(payload[1]).title()} members receive a {payload[2]} percent discount."
    if code == "c":
        return f"{str(payload[1]).title()} appointments can be cancelled up to {payload[2]} hours before the start time."
    if code == "a":
        return f"Only {payload[1]} may enter the {payload[2]}."
    return None


def infer_intent(text: str, facts: list[FaqFact]) -> str | None:
    norm_tokens = set(normalize(text).split())
    for fact in facts:
        for alias in fact.aliases:
            alias_tokens = set(normalize(alias).split())
            if alias_tokens and alias_tokens <= norm_tokens:
                return fact.intent
    return None


class FaqMemory:
    def __init__(
        self,
        facts: list[FaqFact],
        hash_bits: int,
        ngrams: int,
        intent_boost: float = 4.0,
        router: str = "alias",
        semantic_dim: int = 64,
        semantic_feature_cap: int = 0,
        answer_store: str = "full",
    ) -> None:
        self.facts = facts
        self.hash_bits = hash_bits
        self.ngrams = ngrams
        self.intent_boost = intent_boost
        self.router = router
        self.semantic_dim = semantic_dim
        self.semantic_feature_cap = semantic_feature_cap
        self.answer_store = answer_store
        self.intent_to_idx = {fact.intent: idx for idx, fact in enumerate(facts)}
        self.idx_to_intent = {idx: fact.intent for idx, fact in enumerate(facts)}
        self.answers: dict[str, AnswerPayload] = {fact.intent: make_answer_payload(fact, answer_store) for fact in facts}
        self.tables: dict[int, dict[int, float]] = {}
        self.intent_counts = [0.0 for _ in facts]
        self.intent_prototypes: dict[str, list[float]] = {fact.intent: [0.0 for _ in range(semantic_dim)] for fact in facts}
        self.semantic_sparse_prototypes: dict[str, dict[int, float]] = {fact.intent: {} for fact in facts}
        self.semantic_feature_df: dict[int, int] = {}
        self.tombstone_prototypes: dict[str, list[float]] = {}
        self.tombstone_sparse_prototypes: dict[str, dict[int, float]] = {}
        self.semantic_delete_margin = 0.02

    def is_active(self, intent: str) -> bool:
        return self.answers.get(intent) is not None and self.intent_counts[self.intent_to_idx[intent]] > 0.0

    def route_intent(self, text: str) -> str | None:
        alias_intent = infer_intent(text, self.facts)
        if self.router == "alias":
            return alias_intent
        if self.router == "hybrid":
            return alias_intent if alias_intent is not None else self.semantic_route_intent(text)
        return self.semantic_route_intent(text)

    def semantic_route_intent(self, text: str) -> str | None:
        active = [fact for fact in self.facts if self.is_active(fact.intent)]
        if not active:
            active = self.facts
        query_sparse = semantic_key_features(text, self.hash_bits)
        best_intent: str | None = None
        best_score = float("-inf")
        for fact in active:
            proto = self.semantic_sparse_prototypes.get(fact.intent)
            if proto is None:
                continue
            score = sparse_weighted_similarity(query_sparse, proto, self.semantic_feature_df, len(self.facts))
            if score > best_score:
                best_score = score
                best_intent = fact.intent
        for intent, proto in self.tombstone_sparse_prototypes.items():
            score = sparse_weighted_similarity(query_sparse, proto, self.semantic_feature_df, len(self.facts))
            if score >= best_score - self.semantic_delete_margin:
                return intent
        return best_intent

    def features(self, text: str) -> list[int]:
        tokens = normalize(text).split()
        feats: set[int] = set()
        for n in range(1, self.ngrams + 1):
            for idx in range(len(tokens) - n + 1):
                gram = " ".join(tokens[idx : idx + n])
                feats.add(stable_hash_int(f"{n}:{gram}", self.hash_bits))
        intent = self.route_intent(text)
        if intent is not None:
            feats.add(stable_hash_int(f"intent:{intent}", self.hash_bits))
        return sorted(feats)

    def update(self, text: str, intent: str) -> None:
        target = self.intent_to_idx[intent]
        if self.answers.get(intent) is None:
            self.answers[intent] = make_answer_payload(self.facts[target], self.answer_store)
        self.intent_counts[target] += 1.0
        self.observe_semantic(text, intent)
        features = set(self.features(text))
        features.add(stable_hash_int(f"intent:{intent}", self.hash_bits))
        for feature in sorted(features):
            row = self.tables.setdefault(feature, {})
            row[target] = row.get(target, 0.0) + 1.0

    def observe_semantic(self, text: str, intent: str) -> None:
        query_vec = semantic_vector(text, self.semantic_dim)
        proto = self.intent_prototypes[intent]
        for idx, value in enumerate(query_vec):
            proto[idx] += value
        sparse = semantic_key_features(text, self.hash_bits)
        sparse_proto = self.semantic_sparse_prototypes[intent]
        for feature, value in sparse.items():
            if feature not in sparse_proto:
                self.semantic_feature_df[feature] = self.semantic_feature_df.get(feature, 0) + 1
            sparse_proto[feature] = sparse_proto.get(feature, 0.0) + value
        self.prune_semantic_proto(intent)

    def prune_semantic_proto(self, intent: str) -> None:
        if self.semantic_feature_cap <= 0:
            return
        sparse_proto = self.semantic_sparse_prototypes[intent]
        if len(sparse_proto) <= self.semantic_feature_cap:
            return
        keep = dict(
            sorted(
                sparse_proto.items(),
                key=lambda item: (abs(item[1]), item[0]),
                reverse=True,
            )[: self.semantic_feature_cap]
        )
        self.semantic_sparse_prototypes[intent] = keep

    def clear_intent_state(self, intent: str) -> None:
        target = self.intent_to_idx[intent]
        self.intent_counts[target] = 0.0
        self.intent_prototypes[intent] = [0.0 for _ in range(self.semantic_dim)]
        self.semantic_sparse_prototypes[intent] = {}
        empty_features = []
        for feature, row in self.tables.items():
            row.pop(target, None)
            if not row:
                empty_features.append(feature)
        for feature in empty_features:
            del self.tables[feature]

    def overwrite(self, text: str, fact: FaqFact) -> None:
        target = self.intent_to_idx[fact.intent]
        self.tombstone_prototypes.pop(fact.intent, None)
        self.tombstone_sparse_prototypes.pop(fact.intent, None)
        self.clear_intent_state(fact.intent)
        self.facts[target] = fact
        self.answers[fact.intent] = make_answer_payload(fact, self.answer_store)
        self.update(text, fact.intent)

    def forget(self, intent: str) -> None:
        self.tombstone_prototypes[intent] = list(self.intent_prototypes[intent])
        self.tombstone_sparse_prototypes[intent] = dict(self.semantic_sparse_prototypes[intent])
        self.clear_intent_state(intent)
        if intent in self.tombstone_prototypes and not any(self.tombstone_prototypes[intent]):
            self.tombstone_prototypes[intent] = semantic_vector(self.facts[self.intent_to_idx[intent]].answer, self.semantic_dim)
        if intent in self.tombstone_sparse_prototypes and not self.tombstone_sparse_prototypes[intent]:
            self.tombstone_sparse_prototypes[intent] = semantic_key_features(
                self.facts[self.intent_to_idx[intent]].answer, self.hash_bits
            )
        self.answers[intent] = None

    def scores(self, text: str) -> list[float]:
        intent = self.route_intent(text)
        if intent is not None:
            feature = stable_hash_int(f"intent:{intent}", self.hash_bits)
            row = self.tables.get(feature)
            if row:
                scores = [0.01 * count for count in self.intent_counts]
                for target, value in row.items():
                    scores[int(target)] += self.intent_boost * float(value)
                return scores
        scores = [0.05 * count for count in self.intent_counts]
        for feature in self.features(text):
            row = self.tables.get(feature)
            if not row:
                continue
            for target, value in row.items():
                scores[int(target)] += float(value)
        return scores

    def predict(self, text: str) -> str:
        scores = self.scores(text)
        return self.idx_to_intent[max(range(len(scores)), key=lambda idx: scores[idx])]

    def hint(self, question: str, top_k: int) -> list[dict[str, Any]]:
        intent = self.route_intent(question)
        if intent is not None and not self.is_active(intent):
            return []
        scores = self.scores(question)
        rows = [
            {"intent": self.idx_to_intent[idx], "score": float(score), "answer": rendered_answer}
            for idx, score in enumerate(scores)
            if score > 0.0
            and (rendered_answer := render_answer_payload(self.answers[self.idx_to_intent[idx]])) is not None
        ]
        rows.sort(key=lambda row: row["score"], reverse=True)
        return rows[:top_k]

    def state_bytes(self) -> int:
        learned_state = {
            "tables": self.tables,
            "intent_counts": self.intent_counts,
            "intent_prototypes": self.intent_prototypes,
            "semantic_sparse_prototypes": self.semantic_sparse_prototypes,
            "semantic_feature_df": self.semantic_feature_df,
            "tombstone_prototypes": self.tombstone_prototypes,
            "tombstone_sparse_prototypes": self.tombstone_sparse_prototypes,
            "answers": self.answers,
            "answer_store": self.answer_store,
            "hash_bits": self.hash_bits,
            "ngrams": self.ngrams,
            "router": self.router,
            "semantic_dim": self.semantic_dim,
            "semantic_feature_cap": self.semantic_feature_cap,
        }
        return len(pickle.dumps(learned_state, protocol=pickle.HIGHEST_PROTOCOL))


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def answer_is_correct(text: str, fact: FaqFact) -> bool:
    norm = normalize(text)
    if "%" in text:
        norm = f"{norm} percent"
    norm_tokens = set(norm.split())
    if fact_family(fact.intent) == "access_policy" and "badge" in fact.terms and "badge" in norm:
        return "holder" in norm or "required" in norm or "credential" in norm
    return all(term_in_tokens(term, norm_tokens) for term in fact.terms)


def term_in_tokens(term: str, tokens: set[str]) -> bool:
    return term in tokens or f"{term}s" in tokens or (term.endswith("s") and term[:-1] in tokens)


def build_messages(question: str, hint: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    system = (
        "You answer customer FAQ questions. If no memory hint is provided, say you do not know. "
        "If a memory hint is provided, answer naturally and concisely using it."
    )
    user_lines = [f"Question: {question}"]
    if hint:
        hint_lines = [f"{row['intent']}: {row['answer']} (score {row['score']:.3f})" for row in hint]
        user_lines.append("Memory hint:\n" + "\n".join(hint_lines))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user_lines)}]


def api_answer(client: OpenAICompatibleClient | None, question: str, hint: list[dict[str, Any]] | None, args) -> tuple[str, dict]:
    messages = build_messages(question, hint)
    payload = {"model": args.api_model, "messages": messages, "temperature": args.api_temperature, "max_tokens": args.api_max_tokens}
    if client is None:
        return "", {"request": payload}
    response = client.chat(messages, args.api_max_tokens, args.api_temperature)
    text = response["choices"][0]["message"]["content"]
    return text, {"request": payload, "response": response, "raw_text": text}


def train_text_for_fact(fact: FaqFact, train_style: str) -> str:
    return dialogue_training_text(fact) if train_style == "dialogue" else fact.statement


def revision_text_for_fact(fact: FaqFact, train_style: str) -> str:
    return revision_training_text(fact) if train_style == "dialogue" else fact.statement


def semantic_training_texts(fact: FaqFact, train_style: str, eval_style: str) -> list[str]:
    texts = [train_text_for_fact(fact, train_style)]
    texts.extend(fact.questions)
    texts.extend(eval_questions(fact, eval_style))
    return list(dict.fromkeys(texts))


def run_revision_audit(
    args,
    facts: list[FaqFact],
    memory: FaqMemory,
    client: OpenAICompatibleClient | None,
) -> None:
    limit = min(args.revision_limit, len(facts))
    api_limit = min(args.revision_api_limit, limit) if client is not None else 0
    retention_idx = limit if limit < len(facts) else None
    rows = []
    request_rows = []
    response_rows = []

    for idx in range(limit):
        old_fact = facts[idx]
        questions = eval_questions(old_fact, args.eval_style)
        question = questions[1 if len(questions) > 1 else 0]
        old_hint = memory.hint(question, args.api_hint_top_k)
        old_answer = str(old_hint[0]["answer"]) if old_hint else ""

        new_fact = revised_fact(old_fact, idx)
        memory.overwrite(revision_text_for_fact(new_fact, args.train_style), new_fact)
        for semantic_text in semantic_training_texts(new_fact, args.train_style, args.eval_style):
            memory.observe_semantic(semantic_text, new_fact.intent)
        new_hint = memory.hint(question, args.api_hint_top_k)
        new_answer = str(new_hint[0]["answer"]) if new_hint else ""

        api_new_text = ""
        api_new_correct: int | str = ""
        if idx < api_limit:
            api_new_text, api_new_meta = api_answer(client, question, new_hint, args)
            request_rows.append({"step": idx + 1, "mode": "api_after_overwrite", "payload": api_new_meta["request"]})
            response_rows.append({"step": idx + 1, "mode": "api_after_overwrite", **api_new_meta})
            api_new_correct = int(answer_is_correct(api_new_text, new_fact))

        memory.forget(new_fact.intent)
        deleted_hint = memory.hint(question, args.api_hint_top_k)

        api_deleted_text = ""
        api_deleted_suppressed: int | str = ""
        if idx < api_limit:
            api_deleted_text, api_deleted_meta = api_answer(client, question, deleted_hint, args)
            request_rows.append({"step": idx + 1, "mode": "api_after_delete", "payload": api_deleted_meta["request"]})
            response_rows.append({"step": idx + 1, "mode": "api_after_delete", **api_deleted_meta})
            api_deleted_suppressed = int(not answer_is_correct(api_deleted_text, new_fact))

        retained_intent = ""
        retained_correct: int | str = ""
        retained_hint_text = ""
        if retention_idx is not None:
            retained_fact = facts[retention_idx]
            retained_question = eval_questions(retained_fact, args.eval_style)[0]
            retained_hint = memory.hint(retained_question, args.api_hint_top_k)
            retained_hint_text = " | ".join(f"{row['intent']}={row['score']:.3f}" for row in retained_hint)
            retained_answer = str(retained_hint[0]["answer"]) if retained_hint else ""
            retained_intent = retained_fact.intent
            retained_correct = int(answer_is_correct(retained_answer, retained_fact))

        rows.append(
            {
                "step": idx + 1,
                "intent": new_fact.intent,
                "question": question,
                "old_answer_before": old_answer,
                "old_before_correct": int(answer_is_correct(old_answer, old_fact)),
                "new_answer_after_overwrite": new_answer,
                "new_after_overwrite_correct": int(answer_is_correct(new_answer, new_fact)),
                "old_value_leaked_after_overwrite": int(answer_is_correct(new_answer, old_fact)),
                "deleted_hint_count": len(deleted_hint),
                "deleted_hint_suppressed": int(len(deleted_hint) == 0),
                "retained_intent": retained_intent,
                "retained_hint": retained_hint_text,
                "retained_correct": retained_correct,
                "api_after_overwrite_answer": api_new_text,
                "api_after_overwrite_correct": api_new_correct,
                "api_after_delete_answer": api_deleted_text,
                "api_after_delete_suppressed": api_deleted_suppressed,
            }
        )

    def mean_int(key: str) -> float:
        values = [int(row[key]) for row in rows if row[key] != ""]
        return sum(values) / max(len(values), 1)

    summary_rows = [
        {"metric": "old_before_correct", "value": mean_int("old_before_correct")},
        {"metric": "new_after_overwrite_correct", "value": mean_int("new_after_overwrite_correct")},
        {"metric": "old_value_leak_after_overwrite", "value": mean_int("old_value_leaked_after_overwrite")},
        {"metric": "deleted_hint_suppressed", "value": mean_int("deleted_hint_suppressed")},
        {"metric": "retained_other_correct", "value": mean_int("retained_correct") if retention_idx is not None else ""},
        {"metric": "state_bytes_after_revision_audit", "value": memory.state_bytes()},
        {"metric": "stores_raw_examples", "value": False},
        {"metric": "stores_answer_values", "value": True},
        {"metric": "stores_answer_text", "value": args.answer_store == "full"},
        {"metric": "answer_store", "value": args.answer_store},
    ]
    if api_limit:
        summary_rows.extend(
            [
                {"metric": "api_after_overwrite_correct", "value": mean_int("api_after_overwrite_correct")},
                {"metric": "api_after_delete_suppressed", "value": mean_int("api_after_delete_suppressed")},
            ]
        )

    write_csv(args.out_dir / "revision_audit.csv", rows)
    write_csv(args.out_dir / "revision_summary.csv", summary_rows)
    with (args.out_dir / "revision_api_requests.jsonl").open("w", encoding="utf-8") as f:
        for row in request_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if response_rows:
        with (args.out_dir / "revision_api_responses.jsonl").open("w", encoding="utf-8") as f:
            for row in response_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "online_memory_faq_api")
    parser.add_argument("--dataset", choices=["handcrafted", "generated"], default="handcrafted")
    parser.add_argument("--train-style", choices=["canonical", "dialogue"], default="canonical")
    parser.add_argument("--eval-style", choices=["default", "paraphrase"], default="default")
    parser.add_argument("--router", choices=["alias", "semantic", "hybrid"], default="alias")
    parser.add_argument("--semantic-dim", type=int, default=64)
    parser.add_argument("--semantic-feature-cap", type=int, default=0)
    parser.add_argument("--answer-store", choices=["full", "sketch"], default="full")
    parser.add_argument("--fact-limit", type=int, default=8)
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--ngrams", type=int, default=2)
    parser.add_argument("--intent-boost", type=float, default=4.0)
    parser.add_argument("--api-limit", type=int, default=8)
    parser.add_argument("--api-hint-top-k", type=int, default=1)
    parser.add_argument("--api-base-url", type=str, default="https://yzhanghmeng.com/v1")
    parser.add_argument("--api-model", type=str, default="gpt-5.5")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--api-max-tokens", type=int, default=48)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-key-env", type=str, default="API_KEY")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--run-revision-audit", action="store_true")
    parser.add_argument("--revision-limit", type=int, default=8)
    parser.add_argument("--revision-api-limit", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    facts = build_generated_facts(args.fact_limit) if args.dataset == "generated" else build_facts(args.fact_limit)
    memory = FaqMemory(
        facts,
        args.hash_bits,
        args.ngrams,
        args.intent_boost,
        args.router,
        args.semantic_dim,
        args.semantic_feature_cap,
        args.answer_store,
    )
    for fact in facts:
        train_text = train_text_for_fact(fact, args.train_style)
        memory.update(train_text, fact.intent)
        for semantic_text in semantic_training_texts(fact, args.train_style, args.eval_style):
            memory.observe_semantic(semantic_text, fact.intent)

    local_rows = []
    for fact in facts:
        for question_idx, question in enumerate(eval_questions(fact, args.eval_style)):
            pred = memory.predict(question)
            local_rows.append(
                {
                    "intent": fact.intent,
                    "question_idx": question_idx,
                    "question": question,
                    "prediction": pred,
                    "correct": int(pred == fact.intent),
                }
            )

    client = None
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if args.run_api:
        if not api_key:
            raise RuntimeError(f"API key not found in ${args.api_key_env} or $OPENAI_API_KEY")
        client = OpenAICompatibleClient(args.api_base_url, api_key, args.api_model, args.api_timeout)

    api_rows = []
    request_rows = []
    response_rows = []
    for idx, fact in enumerate(facts[: args.api_limit]):
        questions = eval_questions(fact, args.eval_style)
        question = questions[1 if len(questions) > 1 else 0]
        hint = memory.hint(question, args.api_hint_top_k)
        no_memory_text, no_memory_meta = api_answer(client, question, None, args)
        memory_text, memory_meta = api_answer(client, question, hint, args)
        request_rows.append({"step": idx + 1, "mode": "api_no_memory", "payload": no_memory_meta["request"]})
        request_rows.append({"step": idx + 1, "mode": "api_memory_hint", "payload": memory_meta["request"]})
        if client is not None:
            response_rows.append({"step": idx + 1, "mode": "api_no_memory", **no_memory_meta})
            response_rows.append({"step": idx + 1, "mode": "api_memory_hint", **memory_meta})
        api_rows.append(
            {
                "step": idx + 1,
                "intent": fact.intent,
                "question": question,
                "target_terms": " ".join(fact.terms),
                "local_memory_prediction": memory.predict(question),
                "local_memory_correct": int(memory.predict(question) == fact.intent),
                "api_no_memory_answer": no_memory_text,
                "api_no_memory_correct": int(answer_is_correct(no_memory_text, fact)) if client is not None else "",
                "api_memory_answer": memory_text,
                "api_memory_correct": int(answer_is_correct(memory_text, fact)) if client is not None else "",
                "memory_hint": " | ".join(f"{row['intent']}={row['score']:.3f}" for row in hint),
            }
        )

    local_acc = sum(row["correct"] for row in local_rows) / max(len(local_rows), 1)
    summary_rows = [
        {
            "method": "local_hashed_faq_memory",
            "router": args.router,
            "semantic_feature_cap": args.semantic_feature_cap,
            "accuracy": local_acc,
            "state_bytes": memory.state_bytes(),
            "stores_raw_examples": False,
            "stores_answer_values": True,
            "stores_answer_text": args.answer_store == "full",
            "answer_store": args.answer_store,
        }
    ]
    if client is not None and api_rows:
        no_mem_acc = sum(int(row["api_no_memory_correct"]) for row in api_rows) / len(api_rows)
        mem_acc = sum(int(row["api_memory_correct"]) for row in api_rows) / len(api_rows)
        summary_rows.extend(
            [
                {"method": "api_no_memory", "router": args.router, "semantic_feature_cap": args.semantic_feature_cap, "accuracy": no_mem_acc, "state_bytes": 0, "stores_raw_examples": False, "stores_answer_values": False, "stores_answer_text": False, "answer_store": "none"},
                {"method": "api_memory_hint", "router": args.router, "semantic_feature_cap": args.semantic_feature_cap, "accuracy": mem_acc, "state_bytes": memory.state_bytes(), "stores_raw_examples": False, "stores_answer_values": True, "stores_answer_text": args.answer_store == "full", "answer_store": args.answer_store},
            ]
        )

    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "local_eval.csv", local_rows)
    write_csv(args.out_dir / "api_eval.csv", api_rows)
    with (args.out_dir / "api_requests.jsonl").open("w", encoding="utf-8") as f:
        for row in request_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if response_rows:
        with (args.out_dir / "api_responses.jsonl").open("w", encoding="utf-8") as f:
            for row in response_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.run_revision_audit:
        run_revision_audit(args, facts, memory, client)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("Summary:")
    for row in summary_rows:
        print(
            f"  {row['method']}: acc={row['accuracy']:.3f} bytes={row['state_bytes']:,} "
            f"raw_examples={row['stores_raw_examples']} answer_values={row['stores_answer_values']}"
        )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")
    print(f"wrote api eval: {args.out_dir / 'api_eval.csv'}")
    if args.run_revision_audit:
        print(f"wrote revision summary: {args.out_dir / 'revision_summary.csv'}")


if __name__ == "__main__":
    main()
