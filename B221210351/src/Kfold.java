import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.*;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

public class Kfold {
    private static final File veriDosya = new File(Kfold.class.getResource("Data.txt").getPath());
    private int k;
    private DataSet fullDataSet;
    private double[] minValues;
    private double[] maxValues;

    public Kfold(int k) throws FileNotFoundException {
        this.k = k;
        this.fullDataSet = loadDataSet();
    }

    private DataSet loadDataSet() throws FileNotFoundException {
        List<DataSetRow> rows = new ArrayList<>();
        Scanner scanner = new Scanner(veriDosya);

        // Min-Max degerlerinin tanimlanmasi
        minValues = new double[]{Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE}; // 3 giriş + 1 çıkış
        maxValues = new double[]{Double.MIN_VALUE, Double.MIN_VALUE, Double.MIN_VALUE, Double.MIN_VALUE};

        // Veri okuma ve min max degerlerinin hesaplanmasi
        while (scanner.hasNextDouble()) {
            double[] input = {scanner.nextDouble(), scanner.nextDouble(), scanner.nextDouble()};
            double[] output = {scanner.nextDouble()};
            rows.add(new DataSetRow(input, output));
            for (int i = 0; i < 3; i++) { // Girişler için
                if (input[i] < minValues[i]) minValues[i] = input[i];
                if (input[i] > maxValues[i]) maxValues[i] = input[i];
            }
            // Çıkış için
            if (output[0] < minValues[3]) minValues[3] = output[0];
            if (output[0] > maxValues[3]) maxValues[3] = output[0];
        }
        scanner.close();

        // min max normalizasyonunun uygulanmasi
        List<DataSetRow> normalizedRows = new ArrayList<>();
        for (DataSetRow row : rows) {
            double[] normalizedInput = new double[3];
            double[] normalizedOutput = new double[1];

            for (int i = 0; i < 3; i++) {
                normalizedInput[i] = (row.getInput()[i] - minValues[i]) / (maxValues[i] - minValues[i]);
            }
            normalizedOutput[0] = (row.getDesiredOutput()[0] - minValues[3]) / (maxValues[3] - minValues[3]);

            normalizedRows.add(new DataSetRow(normalizedInput, normalizedOutput));
        }

        DataSet normalizedDataSet = new DataSet(3, 1);
        for (DataSetRow normalizedRow : normalizedRows) {
            normalizedDataSet.add(normalizedRow);
        }
        return normalizedDataSet;
    }

    public void performKFoldValidation() {
        List<DataSetRow> allData = new ArrayList<>(fullDataSet.getRows());
        Collections.shuffle(allData); // cross validation
        int foldSize = allData.size() / k;

        double totalTrainError = 0.0;
        double totalTestError = 0.0;

        for (int i = 0; i < k; i++) {
            DataSet trainSet = new DataSet(3, 1);
            DataSet testSet = new DataSet(3, 1);

            for (int j = 0; j < allData.size(); j++) {
                if (j >= i * foldSize && j < (i + 1) * foldSize) {
                    testSet.add(allData.get(j));
                } else {
                    trainSet.add(allData.get(j));
                }
            }

            MultiLayerPerceptron nn = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 3,5,1);
            BackPropagation bp = new BackPropagation();
            bp.setLearningRate(0.01);
            bp.setMaxIterations(700);
            bp.setMaxError(0.0001);
            nn.setLearningRule(bp);

            nn.learn(trainSet);

            double trainError = calculateMSE(nn, trainSet);
            double testError = calculateMSE(nn, testSet);

            totalTrainError += trainError;
            totalTestError += testError;

            System.out.printf("Fold %d - Egitim Hatasi: %.6f, Test Hatasi: %.6f%n", i + 1, trainError, testError);
        }

        System.out.printf("Ortalama Egitim Hatasi: %.6f%n", totalTrainError / k);
        System.out.printf("Ortalama Test Hatasi: %.6f%n", totalTestError / k);
    }

    private double calculateMSE(NeuralNetwork<BackPropagation> network, DataSet dataSet) {
        double totalError = 0.0;
        for (DataSetRow row : dataSet) {
            network.setInput(row.getInput());
            network.calculate();
            double[] output = network.getOutput();
            double error = 0.0;
            for (int i = 0; i < output.length; i++) {
                error += Math.pow(output[i] - row.getDesiredOutput()[i], 2);
            }
            totalError += error;
        }
        return totalError / dataSet.size();
    }

}