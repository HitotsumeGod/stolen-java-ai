package main;

import example.AIService;
import example.DataModel;
import java.util.ArrayList;
import java.util.List;

public class Pilot {
    public static void main(String[] args) {
        AIService aiService = new AIService();

        // Creating training data
        List<DataModel> trainingData = new ArrayList<>();
        trainingData.add(new DataModel(0.1, 0.2, 0.3));
        trainingData.add(new DataModel(0.2, 0.3, 0.5));
        trainingData.add(new DataModel(0.3, 0.4, 0.7));
        trainingData.add(new DataModel(0.4, 0.5, 0.9));

        aiService.trainModel(trainingData);

        // Evaluating the model with test data
        DataModel testData = new DataModel(0.5, 0.5, 1.0);
        aiService.evaluateModel(testData);
    }
}
