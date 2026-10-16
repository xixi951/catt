You are checking the intermediate speech of a voice assistant. While it was still working on the user's question it spoke short updates, numbered below in the order heard; afterwards it gave the final answer shown below. Label each update.

Labels:
- OK: nothing in the update is incorrect, and it does not present as settled anything that the final answer contradicts.
- WRONG: the update states something incorrect: a false fact, a wrong value, formula, identification or reasoning step, or a wrong candidate answer. Judge against established facts and, when one is given, the reference answer. A hedged statement ("probably", "it might be") that is wrong still counts as WRONG, except in the TENTATIVE_CHANGED case below.
- REVERSED: the update presents an answer, result or claim as settled (not hedged), and this turn's own final answer contradicts it. Judge REVERSED only against the final answer of this turn, never against the reference answer.
- TENTATIVE_CHANGED: the update mentions a candidate for the final answer itself (an answer option, or the final number or result asked for) that is explicitly marked as tentative, and the final answer is a different one. This is allowed and is not an error. It never applies to intermediate facts, values, formulas or identifications: a hedged but wrong intermediate statement is WRONG.
If an update fits both WRONG and REVERSED, use WRONG. Saying what is being done ("now checking the second step", "applying conservation of energy") without stating a result is OK. Restating the question is OK.

Question:
<<<
{question}
>>>

Reference answer: {reference}

Final answer as spoken: {final}

Updates:
{numbered}

Reply with JSON only, no other text: {{"labels": [{{"i": 1, "label": "OK" or "WRONG" or "REVERSED" or "TENTATIVE_CHANGED", "note": "at most 20 words"}}, one entry per update, in order]}}
