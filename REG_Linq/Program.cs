using System;
using System.Linq;

namespace LinearRegressionFromScratch
{
    class Program
    {
        static void Main(string[] args)
        {
            float[] X = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            float[] y = { 6, 6, 11, 17, 16, 20, 23, 23, 29, 33, 39 };

            var linearRegressor = new LinearRegressor();
            linearRegressor.Fit(X, y);

            var predictions = linearRegressor.Predict(X);

            Console.WriteLine("Predictions:");
            Console.WriteLine($"{string.Join(", ", predictions.Select(p => p.ToString()))}");

            Console.WriteLine("Actual Value:");
            Console.WriteLine($"{string.Join(", ", y.Select(p => p.ToString()))}");
            Console.ReadLine();
           
        }
    }
    public class LinearRegressor
    {
        private float _b0;
        private float _b1;

        public LinearRegressor()
        {
            _b0 = 0;
            _b1 = 0;
        }

        /// <summary>
        /// Train Linear Regression algoritm.
        /// </summary>
        /// <param name="X">Input Data</param>
        /// <param name="y">Output Data</param>
        public void Fit(float[] X, float[] y)
        {
            var ssxy = X.Zip(y, (a, b) => a * b).Sum() - X.Length * X.Average() * y.Average();
            var ssxx = X.Zip(X, (a, b) => a * b).Sum() - X.Length * X.Average() * X.Average();

            _b1 = ssxy / ssxx;
            _b0 = y.Average() - _b1 * X.Average();
        }

        /// <summary>
        /// Predict new values.
        /// </summary>
        /// <param name="x">Input Data</param>
        /// <returns>Predictions from the trained algoritm.</returns>
        public float[] Predict(float[] x)
        {
            return x.Select(i => _b0 + i * _b1).ToArray();
        }
    }
}