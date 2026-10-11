"""Infrastructure-only generator checks; no numerical solver assessment."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('generator', Path(__file__).parents[1] / 'scripts' / 'generate_stage45a_synthetic.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class GeneratorTests(unittest.TestCase):
    def test_fixed_cases(self):
        self.assertEqual(len(generator.cases()), 9)
        self.assertEqual({x['n'] for x in generator.cases()}, {64, 256, 1024})

    def test_formulas(self):
        for case in generator.cases():
            rows = generator.rows_for(case)
            n = case['n']
            self.assertEqual(len(rows), n)
            for i, row in enumerate(rows):
                self.assertEqual(row[i] - sum(abs(x) for j, x in row.items() if j != i), .5)
                self.assertTrue(all(0 <= j < n and x != 0 for j, x in row.items()))
            self.assertTrue(any(rows[j].get(i, 0) != value for i, row in enumerate(rows)
                                for j, value in row.items() if i != j))
            first = generator.matrix_bytes(case)
            self.assertEqual(first, generator.matrix_bytes(case))
            self.assertTrue(first.startswith(b'%%MatrixMarket matrix coordinate real general\n'))
            if case['family'] == 'arrowhead':
                self.assertEqual(len(rows[0]), case['long_row_degree'] + 1)
            if case['family'] == 'stencil':
                side = case['side']
                self.assertEqual(sum(map(len, rows)), n + 4 * side * (side - 1))


if __name__ == '__main__':
    unittest.main()
