import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

public class BPshowepoch {
	private static final File file = new File(BP.class.getResource("Data.txt").getPath());
	private BackPropagation backpropagation;
	private int maxepoch;
	private double minerror;
	
	public BPshowepoch(double error,int epoch,double learningrate) {
		this.maxepoch = epoch;
	    this.minerror = error;
	    backpropagation = new BackPropagation();
	    backpropagation.setLearningRate(learningrate);
	    backpropagation.setMaxIterations(maxepoch);
	    backpropagation.setMaxError(minerror);
	}	
	public void egitEpochGoster() throws FileNotFoundException {
	    List<DataSetRow> datas = generateTrainingDataSet(file);
	
	    List<DataSetRow> traiData= new ArrayList<>();
	    List<DataSetRow> testData = new ArrayList<>();
	    Random random = new Random();
	
	    for (int i = 0; i < datas.size(); i++) 
	    {
	    	if (random.nextDouble() < 0.75) {
	    		traiData.add(datas.get(i));
	        } 
	    	else 
	    	{
	            testData.add(datas.get(i));
	        }
	    }
	    traiData = normalizeData(traiData);
	    testData = normalizeData(testData);
  
	    DataSet traindataset = new DataSet(3, 1);
	    for (DataSetRow row : traiData) {
	        traindataset.add(row);
	    }
	
	    MultiLayerPerceptron nn = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 3, 5, 1);
	    BackPropagation bp = new BackPropagation();
	    bp.setLearningRate(0.01); 
	    nn.setLearningRule(bp);
	    System.out.println("Epoch basladi \n");
	    	for (int epoch = 1; epoch <= maxepoch; epoch++) {
	            bp.doOneLearningIteration(traindataset); 	        
	            double egitimHatasi = calculateMSE(nn, traiData);
	            double testHatasi = calculateMSE(nn, testData);
	            System.out.println("Epoch: " + epoch + " | Egitim Hatası -> " + egitimHatasi + "  Test Hatasi ->  " + testHatasi);
	            if (bp.getTotalNetworkError() <= minerror) {
	                System.out.println("\nHedef hata değerine ulaşıldı");
	                break;
	            }
	        }
	        nn.save("epochnn.nnet");
	        System.out.println("\nEgitim tamamlandı");
		}
	
    public double calculateMSE(NeuralNetwork<BackPropagation> network, List<DataSetRow> dataSet) {
        return dataSet.stream()
                      .mapToDouble(row -> {
                          network.setInput(row.getInput());
                          network.calculate();
                          double predictedOutput = network.getOutput()[0];
                          double actualOutput = row.getDesiredOutput()[0];
                          return Math.pow(predictedOutput - actualOutput, 2);
                      })
                      .average()
                      .orElse(0.0);
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
                    String[] values = line.split(" "); // Assuming comma-separated values

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

}

