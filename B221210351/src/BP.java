import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.*;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.nnet.MultiLayerPerceptron;

public class BP {
    private static final File file = new File(BP.class.getResource("Data.txt").getPath());
    private BackPropagation backpropagation;
    private int maxepoch;
    private double minerror;

    public BP(double error, int epoch, double learningrate) {
        this.maxepoch = epoch;
        this.minerror = error;
        backpropagation = new BackPropagation();
        backpropagation.setLearningRate(learningrate);
        backpropagation.setMaxIterations(maxepoch);
        backpropagation.setMaxError(minerror);
    }

    public void train() throws FileNotFoundException {
        List<DataSetRow> dataSet = generateTrainingDataSet(file);

        // Veri kontrolü
        if (dataSet.isEmpty()) {
            throw new IllegalStateException("Veri seti boş. Data.txt dosyasını kontrol edin.");
        }
        System.out.println("Toplam veri sayısı: " + dataSet.size());

        List<DataSetRow> trainData = new ArrayList<>();
        List<DataSetRow> testdata = new ArrayList<>();

        // Veriyi eğitim ve test setlerine bölme
        for (int i = 0; i < dataSet.size(); i++) {
            if (i % 4 == 0) { // Her 4 veriden biri test verisi
                testdata.add(dataSet.get(i));
            } else {
                trainData.add(dataSet.get(i));
            }
        }

        // Eğitim ve test verisi kontrolü
        if (trainData.isEmpty() || testdata.isEmpty()) {
            throw new IllegalStateException("Eğitim veya test verisi boş. Veri bölme işlemini kontrol edin.");
        }
        System.out.println("Eğitim verisi boyutu: " + trainData.size());
        System.out.println("Test verisi boyutu: " + testdata.size());

        trainData = normalizeData(trainData);
        testdata = normalizeData(testdata);

        DataSet trainDataSet = new DataSet(3, 1);
        for (DataSetRow row : trainData) {
            trainDataSet.add(row);
        }

        NeuralNetwork<BackPropagation> neuralNetwork = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 3, 5, 1);
        neuralNetwork.setLearningRule(backpropagation);

        System.out.println("Eğitim başlatılıyor");
        neuralNetwork.learn(trainDataSet);
        neuralNetwork.save("nn.nnet");
        System.out.println("Eğitim bitti");

        double trainerror = calculateMSE(neuralNetwork, trainData);
        double testerror = calculateMSE(neuralNetwork, testdata);

        System.out.println("Eğitim Hatası -> " + trainerror);
        System.out.println("Test Hatası -> " + testerror);
    }

    public List<DataSetRow> normalizeData(List<DataSetRow> dataSet) {
        int inputCount = 3;
        int outputCount = 1;
        int totalFeatures = inputCount + outputCount;

        double[] minValues = new double[totalFeatures];
        double[] maxValues = new double[totalFeatures];

        Arrays.fill(minValues, Double.MAX_VALUE);
        Arrays.fill(maxValues, Double.MIN_VALUE);

        for (DataSetRow dataRow : dataSet) {
            double[] inputs = dataRow.getInput();
            double[] outputs = dataRow.getDesiredOutput();

            for (int i = 0; i < inputCount; i++) {
                minValues[i] = Math.min(minValues[i], inputs[i]);
                maxValues[i] = Math.max(maxValues[i], inputs[i]);
            }

            for (int i = 0; i < outputCount; i++) {
                int outputIndex = inputCount + i;
                minValues[outputIndex] = Math.min(minValues[outputIndex], outputs[i]);
                maxValues[outputIndex] = Math.max(maxValues[outputIndex], outputs[i]);
            }
        }

        List<DataSetRow> normalizedDataSet = new ArrayList<>();
        for (DataSetRow dataRow : dataSet) {
            double[] inputs = dataRow.getInput();
            double[] outputs = dataRow.getDesiredOutput();

            double[] normalizedInputs = new double[inputCount];
            double[] normalizedOutputs = new double[outputCount];

            for (int i = 0; i < inputCount; i++) {
                normalizedInputs[i] = normalizeValue(inputs[i], minValues[i], maxValues[i]);
            }

            for (int i = 0; i < outputCount; i++) {
                int outputIndex = inputCount + i;
                normalizedOutputs[i] = normalizeValue(outputs[i], minValues[outputIndex], maxValues[outputIndex]);
            }

            normalizedDataSet.add(new DataSetRow(normalizedInputs, normalizedOutputs));
        }

        return normalizedDataSet;
    }

    private double normalizeValue(double value, double min, double max) {
        return (value - min) / (max - min);
    }

    public List<DataSetRow> generateTrainingDataSet(File inputFile) throws FileNotFoundException {
        try (Scanner scanner = new Scanner(inputFile)) {
            List<DataSetRow> dataSet = new ArrayList<>();

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (!line.isEmpty()) {
                    String[] values = line.split("\\s+"); // Boşluklarla ayrılmış değerleri ayır

                    if (values.length >= 4) { // Ensure enough data for inputs and output
                        double[] inputs = new double[3];
                        double[] outputs = new double[1];

                        for (int i = 0; i < 3; i++) {
                            inputs[i] = Double.parseDouble(values[i]);
                        }
                        outputs[0] = Double.parseDouble(values[3]);

                        dataSet.add(new DataSetRow(inputs, outputs));
                    }
                }
            }

            return dataSet;
        }
    }

    public double calculateMSE(NeuralNetwork<BackPropagation> network, List<DataSetRow> dataSet) {
        double toplamHata = 0;

        for (DataSetRow row : dataSet) {
            network.setInput(row.getInput());
            network.calculate();
            double[] cikti = network.getOutput();
            double[] beklenen = row.getDesiredOutput();

            if (beklenen.length != cikti.length) {
                throw new IllegalArgumentException("Beklenen ve çıktı uzunlukları eşleşmiyor.");
            }

            for (int i = 0; i < beklenen.length; i++) {
                double hata = beklenen[i] - cikti[i];
                toplamHata += hata * hata;
            }
        }

        if (dataSet.isEmpty()) {
            throw new IllegalArgumentException("DataSet boş. Hata hesaplanamaz.");
        }

        return toplamHata / dataSet.size();
    }
}
