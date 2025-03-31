package example;

public class DataModel {
    private double feature1;
    private double feature2;
    private double label;

    public DataModel(double feature1, double feature2, double label) {
        this.feature1 = feature1;
        this.feature2 = feature2;
        this.label = label;
    }
    
    public double getFeature1() {
    	
    	return feature1;
    	
    }
    
    public double getFeature2() {
    	
    	return feature2;
    	
    }
    
    public double getLabel() {
    	
    	return label;
    	
    }

}
