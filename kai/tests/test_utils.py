import unittest

import kai


# NOTE: Dummy test. Use test in this workshop mostly to check that imports
# work as expected on participant's computers.
class TestPerceptron(unittest.TestCase):
    def test_call(self):
        out = kai.perceptron([1, 2, 3, 4, 5, 6])
        self.assertEqual(out, 0)