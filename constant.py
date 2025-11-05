class const:
    #DATASET constants
    DATA_PATH = '/home/cse/msr/csy227518/scratch/Datasets/action_genome'
    SAVE_PATH = 'save_models/multilayer_run'
    FRAMES = 'frames'
    ANNOTATIONS = 'annotations'
    PERSON_BOUNDING_BOX_PKL = 'person_bbox.pkl'
    OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL = 'object_bbox_and_relationship.pkl'
    OBJECT_CLASSES_FILE = 'object_classes.txt'
    BACKGROUND = '__background__'

    #METADATA
    BOUNDING_BOX = 'bbox'
    CLASS = 'class'
    VISIBLE = 'visible'
    METADATA = 'metadata'
    SET = 'set'

    #TRAINING
    NUM_EPOCHS = 25  # Increased for better convergence
    BATCH_SIZE = 32   # Increased for more stable training
    LR = 1e-4        # Increased from 1e-5 for faster learning
