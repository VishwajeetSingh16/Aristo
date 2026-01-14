import nltk

def sentence_score(sentence, query_words):
    sentence_words = set(sentence.lower().split())
    overlap = len(sentence_words & query_words)
    return overlap / (len(sentence_words) + 1)


def is_redundant(sentence, used_sentences):
    for used in used_sentences:
        overlap = len(set(sentence.split()) & set(used.split()))
        if overlap / len(sentence.split()) > 0.6:
            return True
    return False


def format_answer(body):
    return f"""
1. Definition and Overview:
{body[:300]}

2. Detailed Explanation:
{body}

3. Academic Significance:
This topic plays an important role in understanding the subject and is widely applied in academic and practical domains as described in the uploaded textbook.

4. Conclusion:
The above answer strictly follows the provided academic material and explains the concept in a structured and exam-oriented manner.
"""



def build_answer(retrieved_chunks, query, min_words=500, max_words=800):
    query_words = set(query.lower().split())
    ranked_sentences = []

    for item in retrieved_chunks:
        sents = nltk.sent_tokenize(item["chunk"])
        for s in sents:
            # Ignore very short or weak academic sentences
            if len(s.split()) < 8:
                continue

            relevance = sentence_score(s, query_words)

            # UPDATED SCORING (question relevance dominates)
            final_score = (item["score"] * 0.4) + (relevance * 0.6)

            ranked_sentences.append((s, final_score))

    ranked_sentences.sort(key=lambda x: x[1], reverse=True)

    answer = ""
    word_count = 0
    used_sentences = []

    for sentence, _ in ranked_sentences:
        if is_redundant(sentence, used_sentences):
            continue

        words = sentence.split()
        if word_count + len(words) <= max_words:
            answer += sentence + " "
            used_sentences.append(sentence)
            word_count += len(words)

        if word_count >= min_words:
            break

    return format_answer(answer.strip())

