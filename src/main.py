from mlpipeline import MlPipeline


def main():
    try:
        # Initialize and run pipeline
        config_file = "src/config.yaml"
        pipeline = MlPipeline(config_file)

        results = pipeline.run()
        print(f"\nPipeline results: {results}")
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    main();
