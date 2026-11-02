import copy
import json
import unittest

from continual.external_memory import bounded_prompt, chat, retrieve


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def apply_chat_template(self, messages, **kwargs):
        return ''.join(x['role']+':'+x['content']+'\n' for x in messages)+'assistant:'


def example(op, i):
    return {'id':str(i), 'skill':'unused-label', 'prompt':json.dumps({'rows':[{'x':1}], 'ops':[{'op':op}]}),
            'answer':[{'x':i}]}


class ExternalMemoryTests(unittest.TestCase):
    def test_composite_retrieval_covers_operations_without_execution(self):
        memory=[example('filter',i) for i in range(5)]+[example('sort',9),example('join',10)]
        query=json.dumps({'rows':[{'x':3}], 'ops':[{'op':'filter'},{'op':'sort'},{'op':'join'}]})
        chosen=retrieve(query,memory,4)
        self.assertEqual(len(chosen),4)
        self.assertTrue({'filter','sort','join'} <= {json.loads(x['prompt'])['ops'][0]['op'] for x in chosen})

    def test_selection_ignores_answers_and_task_labels(self):
        memory=[example('sort',i) for i in range(8)]
        changed=copy.deepcopy(memory)
        for x in changed:
            x['answer']='a deliberately different answer'
            x['skill']='future-evaluation-label'
        query=memory[0]['prompt']
        self.assertEqual([x['id'] for x in retrieve(query,memory)], [x['id'] for x in retrieve(query,changed)])
        self.assertEqual(retrieve('A general question',memory),[])

    def test_prompt_budget_drops_whole_demonstrations_preserves_query(self):
        tok=Tokenizer();memory=[example('sort',i) for i in range(8)]
        item={'system':'Execute the program.', 'prompt':memory[0]['prompt'], 'answer':'DO NOT LEAK THIS'}
        base=len(chat(tok,item,[]));cap=base+200
        text,chosen,n=bounded_prompt(tok,item,memory,4,cap)
        self.assertLessEqual(n,cap)
        self.assertLess(len(chosen),4)
        self.assertIn(item['prompt'],text)
        self.assertNotIn(item['answer'],text)
        with self.assertRaises(RuntimeError):bounded_prompt(tok,item,memory,4,base-1)


if __name__=='__main__':unittest.main()
