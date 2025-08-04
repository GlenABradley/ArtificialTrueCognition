"""Script to show the semantic space visualization interactively."""
from demo_sensitivity import create_test_field
from visualize_semantics import SemanticVisualizer

def main():
    # Create a test semantic field
    print("Creating test semantic field...")
    field = create_test_field()
    
    # Create visualizer
    visualizer = SemanticVisualizer(field)
    
    # Show initial state
    print("\nShowing initial state... (close the window to continue)")
    visualizer.plot_semantic_space()
    
    # Tune sensitivities
    print("\nTuning sensitivities...")
    field.tune_sensitivities()
    print(f"Adjusted sensitivities: {field.sensitivities.numpy().round(3)}")
    
    # Show state after tuning
    print("\nShowing state after sensitivity tuning...")
    visualizer.plot_semantic_space()
    
    print("\nVisualization complete. Close the window to exit.")

if __name__ == "__main__":
    main()
