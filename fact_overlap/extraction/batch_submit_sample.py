import jsonlines
import csv

# Process each row, write to the new file
inputs = []
with open("../data/aggrefact/aggre_fact.csv", mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        inputs.append(row['summary'].strip())

unique_text = set()

with jsonlines.open('submission4turbo.jsonl', mode='w') as writer:
    for idx, text in enumerate(inputs):
        message = [
            {'role': 'system', 'content': """# Task
1. Extract all atomic facts from the given text and list them in a markdown table with columns: subject, predicate, object direct, object indirect, short adverbial and complement.

2. Extract relations between the events, with columns: discourse1, connective, discourse2. The relations are frequently connected by the following groups of connectives: concession (eg, although, even if); reason (eg, due to, because); result (eg, consequently, therefore); contrast (eg., in contrast, however); disjunctive (eg, otherwise, unless); alternative (eg., instead, as an alternative); condition (eg., provided that, in case); synchronous (eg, when, at the same time); precedence (eg, before, until); succession (eg, after, subsequently); similarity (eg, similarly, in the same way); location (eg, where). The connectives shouldn't be limited by the examples. 

Example text: Being thankful about the exam's results, person A gave a gift to her best friend, person B, last week. Person A would have failed, if person B were not helping. Person B said that her best friend did not need to do this, even if person B helped person A revise all the course materials from time to time. They held a party until midnight. As a result, they didn't get up early on the next day.

Expected output Format:
# Atomic Facts
| Subject | Predicate | Object Direct | Object Indirect | Short Adverbial and Complement |
| - | - | - | - | - | - |
| person A | gave | a gift | to person A's best friend, person B | - | last week |
| person A | was thankful about | the exam's results | - | - | - |
| person B | is best friend of | person A | - | - | - |
| person B | said | her best friend did not need to give a gift to person B | - | - | - |
| person B | helped | person A | - | from time to time | to revise all the course materials |
| person A and person B | held | a party | - | until midnight |
| person A and person B | didn't get up | -  | - | early on the next day |

# Discourse Relations
| Event1 | Connective | Event2 |
| - | - | - |
| person A gave a gift to person A's best friend, person B | because of | being thankful about the exam's results |
| person A would have failed | if | person B were not helping |
| person B said person A did not need to give a gift to person B | even if | person B helped person A revise all the course materials from time to time |
| person A and person B held a party until midnight | As a result | person A and person B didn't get up early on the next day |

# Standards

Having correct tense for predicate is important.

Do not miss any detailed information, especially features, or identities of the entities.

Do not extract content that is not present in the input.

Perform co-reference resolution on entities: if an entity has a full name, put the full name, rather than a pronoun.

If an auxiliary verb refers to another predicate, put their original predicate. For example, "did not need to do this" in the original text is extracted to be "did not need to give a gift to person B".

Make sure the predicate links subject and object in a correct direction. For example, we only know "person B | helped | person A" from the text, rather than "person A | helped | person B". 

If the subject or object has another atomic fact in its adjective, clause, non-finite verb, or appositive, place all related words in the corresponding column and the new fact should also be in another line of the table. For example, "person B | is best friend of | person A" is a new fact inside the appositive "person A's best friend, person B". 

If two discourses are linked with a connective, connect them in the discourse relation table. For example, the events "person A | gave | a gift | to person A's best friend, person B | last week" and "person A | was thankful about | the exam's results" are connected with "because of". The discourse connective shouldn't appear in "Short Adverbial and Complement". 

If a discourse is hypothetical or counterfactual situation, such as subjunctive mood, output the discourse relation but don't extract it as an atomic fact. For example, "person B were not helping" and "person A would have failed" are not atomic facts, but discourse relation should contain "person A would have failed | if | person B were not helping". 

Don't output atomic facts without a predicate.

Discourse connectives should correctly lead another event. Example: "As a result" is leading event the content "person A and person B didn't get up early the next day", rather than anything else.

Don't generate discourse relation table when there is no discourse relations.

Don't output the following connectives: simple conjunction (eg, and, in addition); specification (eg, especially, in particular); restatement (eg, in other words); instantiation (eg, for example, for instance); generalisation (eg, in summary, overall).

Don't output extra explanations"""},
            {'role': 'user', 'content': f"# Context:\n```{text}```"}
        ]
        text = text.strip()
        if text not in unique_text:
            data = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "temperature": 0.3,
                    "n": 5,
                    "logprobs": True,
                    "messages": message,
                }
            }
            unique_text.add(text)
            writer.write(data)
