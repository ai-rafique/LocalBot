"""Unit tests for the pure logic: chunking, keyword tokens, faithfulness,
prompt building and experiment scoring. No Ollama needed.

    python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LOCALBOT_DATA", tempfile.mkdtemp(prefix="localbot-test-"))

from localbot import documents, experiments, generation  # noqa: E402
from localbot.index import keyword_tokens  # noqa: E402


class ParagraphTests(unittest.TestCase):
    def test_numbered_headings_follow_the_document_numbering(self):
        lines = ["1. Setup", "Plug it in.", "2. Usage", "Run it.", "1. A list item inside usage", "3. Limits", "3.1 Power", "Max 5 W."]
        paras = documents._check_heading_numbers(documents._lines_to_paragraphs(lines))
        headings = [p["text"] for p in paras if p["level"]]
        self.assertEqual(headings, ["1. Setup", "2. Usage", "3. Limits", "3.1 Power"])

    def test_table_rows_stay_on_their_own_lines(self):
        paras = documents._lines_to_paragraphs(["Intro text.", "CMD    LEN    CHK", "0x02   2      0x03"])
        self.assertIn("CMD    LEN    CHK\n0x02   2      0x03", [p["text"] for p in paras])

    def test_markdown_headings_and_code_blocks(self):
        paras = documents._markdown_paragraphs("# Title\n\nText here.\n\n```\ncode\nmore\n```\n")
        self.assertEqual(paras[0]["level"], 1)
        self.assertTrue(any(p["text"].startswith("```") and "more" in p["text"] for p in paras))


class ChunkingTests(unittest.TestCase):
    def paras(self):
        out = []
        for s in range(1, 4):
            out.append({"text": f"{s}. Section {s}", "page": s, "level": 1})
            out += [{"text": f"Paragraph {s}.{i} " + "word " * 40, "page": s, "level": 0} for i in range(4)]
        return out

    def test_chunks_respect_size_and_record_sections(self):
        chunks = documents.chunk_paragraphs(self.paras(), 500, 80)
        self.assertTrue(all(len(c["text"]) <= 500 + 80 for c in chunks))
        self.assertTrue(all(c["section"] for c in chunks))
        self.assertEqual({c["page_start"] for c in chunks}, {1, 2, 3})

    def test_a_new_section_starts_a_new_chunk(self):
        chunks = documents.chunk_paragraphs(self.paras(), 2000, 0)
        self.assertEqual(len(chunks), 3)

    def test_overlap_must_be_smaller(self):
        with self.assertRaises(ValueError):
            documents.chunk_paragraphs(self.paras(), 100, 100)


class KeywordTests(unittest.TestCase):
    def test_identifiers_are_kept_whole_and_split(self):
        toks = keyword_tokens("Call Foo::BAR_BAZ with baudRate 0x1A3")
        for t in ("foo::bar_baz", "bar", "baz", "baudrate", "baud", "rate", "0x1a3"):
            self.assertIn(t, toks)
        self.assertNotIn("with", toks)


class FaithfulnessTests(unittest.TestCase):
    hits = [{"source": "My_Doc.pdf", "section": "", "text": "Ratio 12 /34 for ABCD- 5 at 0x1A3; baud 115200."}]

    def test_formatting_differences_are_not_flagged(self):
        c = generation.faithfulness("Per My_Doc, ratio 12/34, ABCD-5 at 0x1A3 [1].", self.hits, [], "q")
        self.assertEqual(c["unsupported"], [])
        self.assertTrue(c["cited"])

    def test_invented_values_and_citations_are_flagged(self):
        c = generation.faithfulness("It uses 0x1B7 and 9600 baud with reg_ctrl_2 [3].", self.hits, [], "q")
        self.assertEqual(c["unsupported"], ["0x1B7", "9600", "reg_ctrl_2"])
        self.assertEqual(c["bad_citations"], [3])


class PromptTests(unittest.TestCase):
    def test_passages_are_numbered_with_their_location(self):
        hits = [{"source": "a.pdf", "section": "2 Setup", "page_start": 3, "page_end": 3, "text": "Alpha."},
                {"source": "b.md", "section": "", "page_start": 0, "page_end": 0, "text": "Beta."}]
        p = generation.build_prompt("What?", hits)
        self.assertIn("[1] a.pdf, p. 3 — 2 Setup\nAlpha.", p)
        self.assertIn("[2] b.md\nBeta.", p)
        self.assertTrue(p.rstrip().endswith("say you don't know."))


class ScoringTests(unittest.TestCase):
    def test_short_tokens_need_word_boundaries(self):
        self.assertFalse(experiments.contains("please accept the change", "cc"))
        self.assertTrue(experiments.contains("use the cc command", "cc"))
        self.assertFalse(experiments.contains("sent 15 frames", "5"))

    def test_spacing_and_separators_are_ignored(self):
        self.assertTrue(experiments.contains("TX: AA 01 00 FF", "aa 01 00 ff"))
        self.assertEqual(experiments.fact_coverage("ttyS1 @ 9600 7E1", [["ttys1"], ["9600"], ["7e1"]]), 1)

    def test_labels(self):
        q = {"answerable": True, "facts": [["0x2a"]]}
        self.assertEqual(experiments.auto_label(q, "It is 0x2A [1].", 1.0), "correct")
        self.assertEqual(experiments.auto_label(q, "I don't know.", 0.0), "missed")
        self.assertEqual(experiments.auto_label({"answerable": False}, "The documents don't state it.", None), "correct")

    def test_question_set_validation(self):
        items, errors = experiments.parse_jsonl('{"id": "a", "question": "q?", "facts": [["x"]]}\n{"id": "b", "question": "q?"}\nnot json')
        self.assertEqual(len(items), 1)
        self.assertEqual(len(errors), 2)


class ClaimSplitTests(unittest.TestCase):
    def test_keeps_checkable_sentences_with_their_citations(self):
        answer = ("Answer:\nThe loader uses port A at 9600 baud [1]. It retries three times before giving up [2].\n"
                  "- The reset command is AA 01 00 FF [1].\n- Ok.\nI don't know what the default is.")
        claims = generation.split_claims(answer)
        self.assertEqual([c["cites"] for c in claims], [[1], [2], [1]])
        self.assertEqual(claims[2]["claim"], "The reset command is AA 01 00 FF.")
        self.assertTrue(all(c["text"] in answer for c in claims))  # needed for highlighting


class CarryOverTests(unittest.TestCase):
    def test_only_identical_answers_to_hand_graded_ones_inherit(self):
        from localbot import storage
        storage.init()
        graded = storage.log_interaction(origin="experiment", question="Q1?", answer="Same answer.")
        storage.query("UPDATE interactions SET grade = 'bad', tags = '[\"wrong\"]', reason = 'r', graded_at = ? WHERE id = ?",
                      (storage.now(), graded))
        run = storage.new_id()
        same = storage.log_interaction(origin="experiment", run_id=run, question="Q1?", answer="Same answer.")
        other = storage.log_interaction(origin="experiment", run_id=run, question="Q1?", answer="Different answer.")
        for qid, iid in (("a", same), ("b", other)):
            storage.query("INSERT INTO run_results VALUES (?, ?, ?, ?)", (run, qid, iid, "{}"))
        self.assertEqual(experiments.carry_over_grades(run), 1)
        row = storage.get_interaction(same)
        self.assertEqual((row["grade"], row["tags"], row["grade_source"]), ("bad", ["wrong"], f"carried:{graded}"))
        self.assertIsNone(storage.get_interaction(other)["grade"])
        # A carried grade is never used as a source for another carry.
        run2 = storage.new_id()
        again = storage.log_interaction(origin="experiment", run_id=run2, question="Q1?", answer="Same answer.")
        storage.query("INSERT INTO run_results VALUES (?, ?, ?, ?)", (run2, "a", again, "{}"))
        experiments.carry_over_grades(run2)
        self.assertEqual(storage.get_interaction(again)["grade_source"], f"carried:{graded}")


if __name__ == "__main__":
    unittest.main()
