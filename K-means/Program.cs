using K_means;
using System;
using System.Globalization;

uint K;
Console.Write("K: ");
K = (uint)int.Parse(Console.ReadLine());
double[][] rawData = LoadData("Final-data.csv", 214, 9, ',');
double[][] data = Normalize(rawData);

KMeans algorithm = new KMeans();
var clusters = algorithm.PartitionToClusters(K, data);
var centroids = algorithm.GetCentroids(clusters);
var wcss = algorithm.CalculateWCSS(centroids);
var bcss = algorithm.CalculateBCSS(centroids);
var dunnIndex = algorithm.CalulateDunnIndex(wcss, bcss);



WriteOutput(clusters);




void WriteOutput(Tuple<uint, double[]>[] output)
{
    //Array.Sort(output, (x, y) => x.Item1.CompareTo(y.Item1));

    using (StreamWriter writer = new StreamWriter("sonuc.txt"))
    {
        int[] sumOfClusters = new int[3];
        for (int i = 0; i < output.Length; i++)
        {
            for (int k = 0; k < K; k++)
            {
                if (output[i].Item1 == k)
                {
                    sumOfClusters[k]++;
                }
            }
            writer.WriteLine($"Kayıt {i}: Küme {output[i].Item1}");
        }
        for (int k = 0; k < K; k++)
        {
            writer.WriteLine($"Küme {k}: {sumOfClusters[k]} kayıt");
        }
        writer.WriteLine("WCSS: " + wcss);
        writer.WriteLine("BCSS: " + bcss);
        writer.WriteLine("Dunn Index: " + dunnIndex);
    }
    Console.WriteLine("Sonuç çıktısı oluşturuldu.");
}

static double[][] Normalize(double[][] rawData)
{
    double[][] result = new double[rawData.Length][];
    for (int i = 0; i < rawData.Length; ++i)
    {
        result[i] = new double[rawData[i].Length];
        Array.Copy(rawData[i], result[i], rawData[i].Length);
    }

    for (int j = 0; j < result[0].Length; ++j) // each col
    {
        double colSum = 0.0;
        for (int i = 0; i < result.Length; ++i)
            colSum += result[i][j];
        double mean = colSum / result.Length;
        double sum = 0.0;
        for (int i = 0; i < result.Length; ++i)
            sum += (result[i][j] - mean) * (result[i][j] - mean);
        double sd = sum / result.Length;
        for (int i = 0; i < result.Length; ++i)
            result[i][j] = (result[i][j] - mean) / sd;
    }
    return result;
}

static double[][] MatrixDouble(int rows, int cols)
{
    double[][] result = new double[rows][];
    for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
    return result;
}

static double[][] LoadData(string fn, int rows, int cols, char delimit)
{
    NumberFormatInfo provider = new NumberFormatInfo();
    provider.NumberDecimalSeparator = ".";
    double[][] result = MatrixDouble(rows, cols);
    FileStream ifs = new FileStream(fn, FileMode.Open);
    StreamReader sr = new StreamReader(ifs);
    string[] tokens = null;
    string line = null;
    int i = 0;
    while ((line = sr.ReadLine()) != null)
    {
        tokens = line.Split(delimit);
        for (int j = 0; j < cols; ++j)
            result[i][j] = Convert.ToDouble(tokens[j], provider);
        ++i;
    }
    sr.Close(); ifs.Close();
    return result;
}



