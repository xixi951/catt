"""Unit tests for speech_norm (EXP-20261009-001 change 5); stdlib unittest."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_norm as N  # noqa: E402


class SplitTest(unittest.TestCase):
    def feed(self, text, accept_comma=False):
        out, buf = [], ''
        for ch in text:                      # character-by-character streaming
            buf += ch
            s, buf = N.split_sentences(buf, accept_comma=accept_comma)
            out += s
        if buf.strip():
            out.append(buf.strip())
        return out

    def test_decimal_not_split(self):
        self.assertEqual(self.feed('The speed is 0.14c here. Next one.'), ['The speed is 0.14c here.', 'Next one.'])
        self.assertEqual(self.feed('So gamma approx 3.6 overall.'), ['So gamma approx 3.6 overall.'])
        self.assertEqual(self.feed('Energy is 5.96 mc squared.'), ['Energy is 5.96 mc squared.'])

    def test_abbrev_not_split(self):
        self.assertEqual(self.feed('The U.S. economy grew, e.g. in 2020. Then it fell.'),
                         ['The U.S. economy grew, e.g. in 2020.', 'Then it fell.'])
        self.assertEqual(self.feed('Dr. Smith said so. Yes.'), ['Dr. Smith said so.', 'Yes.'])

    def test_plain(self):
        self.assertEqual(self.feed('One. Two! Three? Four'), ['One.', 'Two!', 'Three?', 'Four'])

    def test_comma_first_chunk(self):
        self.assertEqual(N.split_sentences('Okay, a question', accept_comma=True)[0], ['Okay,'])
        self.assertEqual(N.split_sentences('About 1,000 people', accept_comma=True)[0], [])

    def test_clauses(self):
        s = 'I am applying conservation of energy, and now working out the speed at the bottom.'
        c = N.split_clauses(s)
        self.assertEqual(len(c), 2)
        self.assertEqual(' '.join(c), s)
        self.assertEqual(N.split_clauses('Okay, so now checking it.'), ['Okay, so now checking it.'])


class SpokenTest(unittest.TestCase):
    def test_latex(self):
        self.assertEqual(N.spoken('The answer is $\\boxed{277}$.'), 'The answer is 277.')
        self.assertIn('1 over 2', N.spoken('It is $\\frac{1}{2}$ here.'))
        self.assertIn('x squared', N.spoken('We get $x^2 + 1$.'))
        self.assertIn('the square root of', N.spoken('$\\sqrt{3}$ is it'))

    def test_long_formula(self):
        f = '$' + ' + '.join(f'a_{{{i}}} x^{{{i}}}' for i in range(12)) + '$'
        info = {}
        out = N.spoken(f'The sum is {f}.', info)
        self.assertIn(N.LONG_FORMULA_PHRASE, out)
        self.assertEqual(info.get('long_formula'), 1)

    def test_prose_unchanged(self):
        t = 'A well-known result, from the 19th century, says so.'
        self.assertEqual(N.spoken(t), t)

    def test_markdown(self):
        self.assertEqual(N.spoken('**Answer:** B'), 'Answer: B')

    def test_never_raises(self):
        self.assertIsInstance(N.spoken('\\frac{'), str)


class KindTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(N.question_kind('Solve. Put your answer inside \\boxed{}.\n\nFind x.'), 'closed')
        self.assertEqual(N.question_kind('Which?\n\nA) one\nB) two\nC) three\nD) four'), 'closed')
        self.assertEqual(N.question_kind('What is collusion?\nA. x\nB. y\nC. z\nD. w'), 'closed')
        self.assertEqual(N.question_kind('How did US states get their names?'), 'open')


if __name__ == '__main__':
    unittest.main()
