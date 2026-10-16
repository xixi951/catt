You are rating sentences from a spoken reply by a voice assistant to a user's question. The sentences are numbered in the order the listener heard them.

Label every sentence Y (substantive) or N (not substantive).

A sentence is substantive if it gives information beyond the question text that bears directly on this question: for example a specific solution approach or method for this question, an intermediate result, a candidate or final answer, or, for an open-ended request, content that starts answering the request.

A sentence is not substantive if it only acknowledges, restates or paraphrases the question (including its given quantities or answer options), announces that thinking or checking is under way without saying what specifically, or is generic filler. A sentence that only names the topic of the question is not substantive.

Judge each sentence on its own content, in the context of the question and the sentences before it. Correctness does not matter here: a wrong but specific approach or answer is still substantive.

Question:
<<<
{question}
>>>

Sentences:
{numbered}

Reply with JSON only, no other text: {{"labels": ["Y" or "N", one entry per sentence, in order]}}
