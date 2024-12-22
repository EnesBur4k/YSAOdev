import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.*;
import org.neuroph.nnet.*;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

public class BPMomentum {
    private static final File dataFile = new File(BPMomentum.class.getResource("Data.txt").getPath());
    private double[] maxValues;
    private double[] minValues;
    private DataSet trainingDataSet;
    private DataSet testDataSet;
    private MomentumBackpropagation backpropagation;
    private int hiddenLayerNeuronCount;

    public BPMomentum(double momentum,int epochs) {
        maxValues = new double[4];
        minValues = new double[4];
        Arrays.fill(maxValues, Double.MIN_VALUE);
        Arrays.fill(minValues, Double.MAX_VALUE);

        calculateMaxValues(dataFile);

        DataSet fullDataSet = loadDataSet(dataFile);
        splitDataSet(fullDataSet, 0.75);

        backpropagation = new MomentumBackpropagation();
        backpropagation.setMomentum(momentum);
        backpropagation.setLearningRate(0.01);
        backpropagation.setMaxError(0.0001);
        backpropagation.setMaxIterations(epochs);
        this.hiddenLayerNeuronCount = 5;
    }

    private void calculateMaxValues(File dataFile) {
        try (Scanner scanner = new Scanner(dataFile)) {
            while (scanner.hasNextLine()) {
                String[] values = scanner.nextLine().trim().split(" ");
                for (int i = 0; i < 3; i++) {
                    double value = Double.parseDouble(values[i]);
                    maxValues[i] = Math.max(maxValues[i], value);
                    minValues[i] = Math.min(minValues[i], value);
                }
                double output = Double.parseDouble(values[3]);
                maxValues[3] = Math.max(maxValues[3], output);
                minValues[3] = Math.min(minValues[3], output);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void train() {
        MultiLayerPerceptron neuralNetwork = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 3, hiddenLayerNeuronCount, 1);
        neuralNetwork.setLearningRule(backpropagation);

        System.out.println("Egitim basliyor");
        neuralNetwork.learn(trainingDataSet);
        System.out.println("Egitim tamamlandi.");

        neuralNetwork.save("mbpnn.nnet");

        System.out.println("Egitim hatasi -> " + calculateMSE(neuralNetwork, trainingDataSet));
        System.out.println("Test hatasi -> " + calculateMSE(neuralNetwork, testDataSet));
    }

    private double calculateMSE(NeuralNetwork neuralNetwork, DataSet dataSet) {
        return dataSet.getRows().stream().mapToDouble(row -> {
            neuralNetwork.setInput(row.getInput());
            neuralNetwork.calculate();
            double[] output = neuralNetwork.getOutput();
            double[] expected = row.getDesiredOutput();
            return Arrays.stream(expected).map(i -> Math.pow(i - output[0], 2)).sum();
        }).sum() / dataSet.size();
    }

    private void splitDataSet(DataSet fullDataSet, double ratio) {
        List<DataSetRow> rows = new ArrayList<>(fullDataSet.getRows());
        Collections.shuffle(rows);
        int trainingSize = (int) (rows.size() * ratio);
        trainingDataSet = new DataSet(3, 1);
        testDataSet = new DataSet(3, 1);
        for (int i = 0; i < rows.size(); i++) {
            if (i < trainingSize) {
                trainingDataSet.add(rows.get(i));
            } else {
                testDataSet.add(rows.get(i));
            }
        }
    }

    public double singleTest(double[] input) {
        NeuralNetwork<?> model = NeuralNetwork.createFromFile("mbpnn.nnet");
        model.setInput(input);
        model.calculate();
        return model.getOutput()[0];
    }

    private DataSet loadDataSet(File dataFile) {
        DataSet dataSet = new DataSet(3, 1);
        try (Scanner scanner = new Scanner(dataFile)) {
            while (scanner.hasNextLine()) {
                String[] values = scanner.nextLine().trim().split(" ");
                double[] inputs = new double[3];
                for (int i = 0; i < 3; i++) {
                    double value = Double.parseDouble(values[i]);
                    inputs[i] = normalizeValue(maxValues[i], minValues[i], value);
                }
                double output = Double.parseDouble(values[3]);
                double[] outputs = {normalizeValue(maxValues[3], minValues[3], output)};
                dataSet.add(new DataSetRow(inputs, outputs));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return dataSet;
    }

    private double normalizeValue(double max, double min, double value) {
        return (max == min) ? 0.0 : (value - min) / (max - min);
    }
    
    private double denormalizeValue(double normalizedValue, double max, double min) {
        return normalizedValue * (max - min) + min;
    }

    public double testValue(double x, double y, double z) {
        // Modeli yükle
        NeuralNetwork<BackPropagation> neuralNetwork = NeuralNetwork.createFromFile("mbpnn.nnet");
        
        // Girdileri ayarla
        double[] inputs = {
            normalizeValue(maxValues[0], minValues[0], x),
            normalizeValue(maxValues[1], minValues[1], y),
            normalizeValue(maxValues[2], minValues[2], z)
        };
        
        neuralNetwork.setInput(inputs);
        neuralNetwork.calculate();
        
        // Normalize edilmiş çıktıyı gerçek değere dönüştür
        double normalizedOutput = neuralNetwork.getOutput()[0];
        return denormalizeValue(normalizedOutput, maxValues[3], minValues[3]);
    }

}