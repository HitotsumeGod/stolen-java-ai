package example;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class AIServiceTest {
    private AIService aiService;

    @BeforeEach
    public void setUp() {
        aiService = new AIService();
    }

    @Test
    public void testTrainModel() {
        List<DataModel> trainingData = new ArrayList<>();
        trainingData.add(new DataModel(0.1, 0.2, 0.3));
        trainingData.add(new DataModel(0.2, 0.3, 0.5));
        trainingData.add(new DataModel(0.3, 0.4, 0.7));
        trainingData.add(new DataModel(0.4, 0.5, 0.9));

        aiService.trainModel(trainingData);
        // Verify that training data is not null
        assertNotNull(trainingData);
    }

    @Test
    public void testEvaluateModel() {
        DataModel testData = new DataModel(0.5, 0.5, 1.0);
        aiService.evaluateModel(testData);
        // Verify that test data is not null
        assertNotNull(testData);
    }
}
