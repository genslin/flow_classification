import argparse
import preprocessing
import models_and_training
import logging_functions as log
import model_evaluation


def get_args():
    parser = argparse.ArgumentParser(
        description="Flow Regime Classifier — Training and Evaluation Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    # ====================================================================
    # train subcommand
    # ====================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model (interactive head/layer4 loops).",
    )

    # --- Dataset Split Arguments ---
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of dataset reserved for final testing (default: 0.15).",
    )

    train_parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of dataset reserved for validation during training (default: 0.15).",
    )

    train_parser.add_argument(
        "--rng",
        type=int,
        default=42,
        help="Random seed used for dataset shuffling and splits (default: 42).",
    )

    # --- Image Preprocessing ---
    train_parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=[224, 224],
        help="Image resize dimensions: WIDTH HEIGHT (default: 224 224).",
    )

    train_parser.add_argument(
        "--random-rotation",
        type=float,
        default=10.0,
        help="Maximum rotation angle for random rotation augmentation (default: 10°).",
    )

    train_parser.add_argument(
        "--brightness-jitter",
        type=float,
        default=0.2,
        help="Brightness jitter strength for color augmentation (default: 0.2).",
    )

    train_parser.add_argument(
        "--contrast-jitter",
        type=float,
        default=0.2,
        help="Contrast jitter strength for color augmentation (default: 0.2).",
    )

    train_parser.add_argument(
        "--normalize-mean",
        nargs=3,
        type=float,
        default=[0.485, 0.456, 0.406],
        help="Normalization mean for each RGB channel (default: 0.485 0.456 0.406).",
    )

    train_parser.add_argument(
        "--normalize-std",
        nargs=3,
        type=float,
        default=[0.229, 0.224, 0.225],
        help="Normalization std dev for each RGB channel (default: 0.229 0.224 0.225).",
    )

    # --- DataLoader Options ---
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of subprocesses used for DataLoader workers (default: 2).",
    )

    train_parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Use pinned memory in data loading (recommended when training on GPU).",
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training, validation, and testing (default: 32).",
    )

    # --- Model Options ---
    train_parser.add_argument(
        "--model-name",
        type=str,
        default="default_resnet18",
        help="Name assigned to the model for logging and saving (default: default_resnet18).",
    )

    train_parser.add_argument(
        "--load-existing",
        action="store_false",
        dest="new_model",
        help="Load existing model weights instead of creating a new model.",
    )

    train_parser.add_argument(
        "--loss-fn-name",
        type=str,
        default="CrossEntropyLoss",
        choices=["CrossEntropyLoss", "NLLLoss"],
        help="Loss function to use: CrossEntropyLoss or NLLLoss (default: CrossEntropyLoss).",
    )

    train_parser.add_argument(
        "--optimizer-name",
        type=str,
        default="Adam",
        choices=["Adam", "SGD"],
        help="Optimizer to use: Adam or SGD (default: Adam).",
    )

    # --- Optimizer Hyperparameters ---
    train_parser.add_argument(
        "--fc-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the fully connected classification head (default: 0.001).",
    )

    train_parser.add_argument(
        "--layer4-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the last ResNet block during fine-tuning (default: 0.0001).",
    )

    train_parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum value used for SGD optimizer (default: 0.9). Ignored for Adam.",
    )

    # ====================================================================
    # test subcommand
    # ====================================================================
    test_parser = subparsers.add_parser(
        "test",
        help="Once training of a model is complete, perform the final evaluation on the test data",
    )
    test_parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the saved model to evaluate.",
    )
    # --- Image Preprocessing ---
    test_parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=[224, 224],
        help="Image resize dimensions: WIDTH HEIGHT (default: 224 224).",
    )

    test_parser.add_argument(
        "--normalize-mean",
        nargs=3,
        type=float,
        default=[0.485, 0.456, 0.406],
        help="Normalization mean for each RGB channel (default: 0.485 0.456 0.406).",
    )

    test_parser.add_argument(
        "--normalize-std",
        nargs=3,
        type=float,
        default=[0.229, 0.224, 0.225],
        help="Normalization std dev for each RGB channel (default: 0.229 0.224 0.225).",
    )

    # --- DataLoader Options ---
    test_parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of subprocesses used for DataLoader workers (default: 2).",
    )

    test_parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Use pinned memory in data loading (recommended when training on GPU).",
    )

    test_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training, validation, and testing (default: 32).",
    )

    # --- Model Options ---
    test_parser.add_argument(
        "--loss-fn-name",
        type=str,
        default="CrossEntropyLoss",
        choices=["CrossEntropyLoss", "NLLLoss"],
        help="Loss function to use: CrossEntropyLoss or NLLLoss (default: CrossEntropyLoss).",
    )

    # ====================================================================
    # plot subcommand
    # ====================================================================
    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot performance data of a model",
    )
    plot_parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the saved model to plot.",
    )

    # ====================================================================
    # autotrain subcommand
    # ====================================================================
    autotrain_parser = subparsers.add_parser(
        "autotrain",
        help="Automatically train a model with the default selections",
    )
    autotrain_parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to be trained.",
    )
    return parser.parse_args()


def run_train(args):
    # ------------------------------------------------------------
    # Load an existing model OR create a new one.
    #   --model-name   = name assigned to this run
    #   --new-model   = false → load existing weights
    #
    # The model object returned always has a .name attribute used
    # for folder naming, logging, performance tracking, etc.
    # ------------------------------------------------------------
    model = models_and_training.load_model(
        model_name=args.model_name, new_model=args.new_model
    )

    # ------------------------------------------------------------
    # Build train/val/test dataloaders with full preprocessing
    # (resize, augmentations, normalization, batching, etc.)
    #
    # This returns 3 PyTorch DataLoader objects ready for training.
    # If loading a saved model the same indices will be used to ensure
    # the model cannot been trained on the test data
    # ------------------------------------------------------------
    if args.new_model:
        train_loader, val_loader, test_loader = (
            preprocessing.get_dataloaders_complete_preprocessing(
                test_size=args.test_size,
                val_size=args.val_size,
                rng=args.rng,
                image_size=tuple(args.image_size),  # convert list → tuple
                random_rotation=args.random_rotation,
                brightness_jitter=args.brightness_jitter,
                contrast_jitter=args.contrast_jitter,
                normalize_mean=args.normalize_mean,
                normalize_std=args.normalize_std,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
        )
    else:
        train_loader, val_loader, test_loader = (
            preprocessing.load_train_val_test_dataloaders(
                model=model,
                image_size=tuple(args.image_size),  # convert list → tuple
                random_rotation=args.random_rotation,
                brightness_jitter=args.brightness_jitter,
                contrast_jitter=args.contrast_jitter,
                normalize_mean=args.normalize_mean,
                normalize_std=args.normalize_std,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
        )

    # ------------------------------------------------------------
    # Load previous training history (if it exists).
    # performance_data is a list of dicts:
    #    [{"epoch": ..., "batch": ..., "loss": ...}, ...]
    #
    # If no CSV exists yet, a new empty list is returned.
    # ------------------------------------------------------------
    performance_data = models_and_training.load_performance_data(model)

    # Determine the next epoch index.
    # If performance_data is empty, start at epoch 0.
    # If it contains past runs, continue from the last epoch + 1.
    if performance_data:
        starting_epoch = performance_data[-1]["epoch"] + 1
    else:
        starting_epoch = 0

    # ------------------------------------------------------------
    # Stage 1: Train only the final fully-connected (fc) layer.
    # We freeze all backbone parameters and unfreeze only fc().
    #
    # This is a standard transfer-learning first step:
    #   1. Freeze backbone (pre-trained features stay intact)
    #   2. Train classification head only
    # ------------------------------------------------------------
    model = models_and_training.freeze_model_parameters(model=model)
    model = models_and_training.unfreeze_fc(model=model)

    # Get the chosen loss function (default: CrossEntropyLoss)
    loss_fn = models_and_training.get_loss_fn(loss_fn_name=args.loss_fn_name)

    # ------------------------------------------------------------
    # INTERACTIVE TRAINING LOOP (Head Only)
    #
    # The user is asked if they want to train the head-only stage.
    # If YES:
    #    - The optimizer is created for the head only
    #    - The user can train multiple rounds of N epochs
    #    - Validation confusion matrices can be plotted after each round
    # ------------------------------------------------------------
    train_head_solo = (
        input("Do you want to train the model head by itself? (Y/N)").strip().upper()
    )

    if train_head_solo == "Y":
        continue_training_head = True
        head_only = True

        # Optimizer for the fully connected head
        optimizer = models_and_training.get_optimizer(
            model=model,
            layer="fc",
            optimizer_name=args.optimizer_name,
            fc_learning_rate=args.fc_learning_rate,
            momentum=args.momentum,
        )

        while continue_training_head:
            # Ask user how many epochs to train for this round
            epochs_to_train = int(input("Number of epochs to train: "))

            # Log the training start
            logger = log.get_model_logger(model_name=model.name)
            logger.info(f"Training model head for {epochs_to_train} epochs")

            # Train the model for the requested number of epochs
            model, performance_data = models_and_training.train_model_for_epochs(
                starting_epoch=starting_epoch,
                number_of_epochs_to_train=epochs_to_train,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                prior_performance_data=performance_data,
                head_only=head_only,
            )

            # Update epoch index for next round
            starting_epoch += epochs_to_train

            # Evaluate on validation set to check progress
            confusion_matrix = model_evaluation.test_loop(
                dataloader=val_loader,
                model=model,
                loss_fn=loss_fn,
            )

            # Optionally plot/save confusion matrix
            plot_matrix = (
                input("Do you want to plot the confusion matrix? (Y/N)").strip().upper()
            )
            if plot_matrix == "Y":
                save_matrix = (
                    input("Do you want to save the plot? (Y/N)").strip().upper()
                )
                if save_matrix == "Y":
                    plt_name = input("Enter the name you want to save the plot with: ")
                    model_evaluation.plot_confusion_matrix(
                        model=model,
                        cm=confusion_matrix,
                        plt_name=plt_name,
                        save_plot=True,
                    )
                else:
                    model_evaluation.plot_confusion_matrix(
                        model=model, cm=confusion_matrix, plt_name="", save_plot=False
                    )

            # Ask user if they want to continue training the head
            more_head_training = input("Continue training head? (Y/N)").strip().upper()
            if more_head_training == "N":
                continue_training_head = False
            # If input is invalid or "Y", just loop again

    # ------------------------------------------------------------
    # INTERACTIVE TRAINING LOOP (Fine-tune layer4 + head)
    #
    # If the user chooses, we unfreeze the last ResNet block (layer4)
    # and jointly fine-tune it + the fc head.
    #
    # Again, this can run for multiple multi-epoch rounds.
    # ------------------------------------------------------------
    train_head_and_layer4 = (
        input("Do you want to train layer 4 and the model head? (Y/N)").strip().upper()
    )

    if train_head_and_layer4 == "Y":
        continue_training_layer4 = True
        head_only = False
        # Unfreeze last ResNet block (layer4)
        model = models_and_training.unfreeze_last_block(model=model)

        # Create optimizer with separate LR for layer4 and fc
        optimizer = models_and_training.get_optimizer(
            model=model,
            layer="layer4",
            optimizer_name=args.optimizer_name,
            fc_learning_rate=args.fc_learning_rate,
            layer4_learning_rate=args.layer4_learning_rate,
            momentum=args.momentum,
        )

        while continue_training_layer4:
            epochs_to_train = int(input("Number of epochs to train: "))
            logger = log.get_model_logger(model_name=model.name)
            logger.info(f"Training model head and layer4 for {epochs_to_train} epochs")

            # Train for this round
            model, performance_data = models_and_training.train_model_for_epochs(
                starting_epoch=starting_epoch,
                number_of_epochs_to_train=epochs_to_train,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                prior_performance_data=performance_data,
                head_only=head_only,
            )

            starting_epoch += epochs_to_train

            # Validate after this round
            confusion_matrix = model_evaluation.test_loop(
                dataloader=val_loader,
                model=model,
                loss_fn=loss_fn,
            )

            # Optional confusion matrix plotting
            plot_matrix = (
                input("Do you want to plot the confusion matrix? (Y/N)").strip().upper()
            )
            if plot_matrix == "Y":
                save_matrix = (
                    input("Do you want to save the plot? (Y/N)").strip().upper()
                )
                if save_matrix == "Y":
                    plt_name = input("Enter the name you want to save the plot with: ")
                    model_evaluation.plot_confusion_matrix(
                        model=model,
                        cm=confusion_matrix,
                        plt_name=plt_name,
                        save_plot=True,
                    )
                else:
                    model_evaluation.plot_confusion_matrix(
                        model=model, cm=confusion_matrix, plt_name="", save_plot=False
                    )

            # Ask user if they want another round of fine-tuning
            more_training = input("Continue training model? (Y/N)").strip().upper()
            if more_training == "N":
                continue_training_layer4 = False

    # ------------------------------------------------------------
    # END OF TRAINING SESSION
    #
    # Ask the user whether to save:
    #   - updated performance_data (as CSV)
    #   - model weights (.pth)
    #
    # This always runs, no matter which training paths they followed.
    # ------------------------------------------------------------
    save_reply_valid = False
    while not save_reply_valid:
        save_model = (
            input("Save model weights and performance data? (Y/N)").strip().upper()
        )

        if save_model == "Y":
            save_reply_valid = True
            models_and_training.save_performance_data(
                performance_data=performance_data, model=model
            )
            models_and_training.save_model(model=model)
            preprocessing.save_train_val_test_indices(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                model=model,
            )

        elif save_model == "N":
            save_reply_valid = True

    print("Session Ended")


def run_test(args):
    model = models_and_training.load_model(model_name=args.model_name, new_model=False)
    test_loader = preprocessing.load_test_dataloader(
        model=model,
        image_size=tuple(args.image_size),  # convert list → tuple
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    model = models_and_training.freeze_model_parameters(model=model)
    # Get the chosen loss function (default: CrossEntropyLoss)
    loss_fn = models_and_training.get_loss_fn(loss_fn_name=args.loss_fn_name)
    logger = log.get_model_logger(model.name)
    logger.info("Running final model evaluation on test data:")
    confusion_matrix = model_evaluation.test_loop(
        dataloader=test_loader,
        model=model,
        loss_fn=loss_fn,
    )
    # Optionally plot/save confusion matrix
    plot_matrix = (
        input("Do you want to plot the confusion matrix? (Y/N)").strip().upper()
    )
    if plot_matrix == "Y":
        save_matrix = input("Do you want to save the plot? (Y/N)").strip().upper()
        if save_matrix == "Y":
            plt_name = input("Enter the name you want to save the plot with: ")
            model_evaluation.plot_confusion_matrix(
                model=model,
                cm=confusion_matrix,
                plt_name=plt_name,
                save_plot=True,
            )
        else:
            model_evaluation.plot_confusion_matrix(
                model=model, cm=confusion_matrix, plt_name="", save_plot=False
            )


def run_plot(args):
    logger = log.get_model_logger(args.model_name)
    logger.info("Loading model to retrieve performance data for plot")
    model = models_and_training.load_model(model_name=args.model_name, new_model=False)
    performance_data = models_and_training.load_performance_data(
        model=model, plot_only=True
    )
    save = input("Do you want to save the plot? (Y/N)").strip().upper()
    if save == "Y":
        save_plot = True
        plot_name = input("Enter the name you want to save the plot as: ")
    else:
        save_plot = False
        plot_name = None
    model_evaluation.plot_performance_data(
        model=model,
        performance_data=performance_data,
        save_plot=save_plot,
        plot_name=plot_name,
    )

def run_autotrain(args):
    # ------------------------------------------------------------
    # Load an existing model OR create a new one.
    #   --model-name   = name assigned to this run
    #   --new-model   = false → load existing weights
    #
    # The model object returned always has a .name attribute used
    # for folder naming, logging, performance tracking, etc.
    # ------------------------------------------------------------
    model = models_and_training.load_model(
        model_name=args.model_name, new_model=True
    )

    # ------------------------------------------------------------
    # Build train/val/test dataloaders with full preprocessing
    # (resize, augmentations, normalization, batching, etc.)
    #
    # This returns 3 PyTorch DataLoader objects ready for training.
    # If loading a saved model the same indices will be used to ensure
    # the model cannot been trained on the test data
    # ------------------------------------------------------------
    train_loader, val_loader, test_loader = (
        preprocessing.get_dataloaders_complete_preprocessing()
    )

    # ------------------------------------------------------------
    # Load previous training history (if it exists).
    # performance_data is a list of dicts:
    #    [{"epoch": ..., "batch": ..., "loss": ...}, ...]
    #
    # If no CSV exists yet, a new empty list is returned.
    # ------------------------------------------------------------
    performance_data = models_and_training.load_performance_data(model)

    # Determine the next epoch index.
    # If performance_data is empty, start at epoch 0.
    # If it contains past runs, continue from the last epoch + 1.
    starting_epoch = 0

    # ------------------------------------------------------------
    # Stage 1: Train only the final fully-connected (fc) layer.
    # We freeze all backbone parameters and unfreeze only fc().
    #
    # This is a standard transfer-learning first step:
    #   1. Freeze backbone (pre-trained features stay intact)
    #   2. Train classification head only
    # ------------------------------------------------------------
    model = models_and_training.freeze_model_parameters(model=model)
    model = models_and_training.unfreeze_fc(model=model)

    # Get the chosen loss function (default: CrossEntropyLoss)
    loss_fn = models_and_training.get_loss_fn()

    # ------------------------------------------------------------
    # INTERACTIVE TRAINING LOOP (Head Only)
    #
    # The user is asked if they want to train the head-only stage.
    # If YES:
    #    - The optimizer is created for the head only
    #    - The user can train multiple rounds of N epochs
    #    - Validation confusion matrices can be plotted after each round
    # ------------------------------------------------------------

    head_only = True

    # Optimizer for the fully connected head
    optimizer = models_and_training.get_optimizer(
        model=model,
        layer="fc",
    )

    epochs_to_train = 10

    # Log the training start
    logger = log.get_model_logger(model_name=model.name)
    logger.info(f"Training model head for {epochs_to_train} epochs")

    # Train the model for the requested number of epochs
    model, performance_data = models_and_training.train_model_for_epochs(
        starting_epoch=starting_epoch,
        number_of_epochs_to_train=epochs_to_train,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        prior_performance_data=performance_data,
        head_only=head_only,
        save_confusion_matrix=True
    )

    # Update epoch index for next round
    starting_epoch += epochs_to_train
    head_only = False

    # Unfreeze last ResNet block (layer4)
    model = models_and_training.unfreeze_last_block(model=model)

    # Create optimizer with separate LR for layer4 and fc
    optimizer = models_and_training.get_optimizer(
        model=model,
        layer="layer4",
    )

    epochs_to_train = 10
    logger.info(f"Training model head and layer4 for {epochs_to_train} epochs")

    # Train for this round
    model, performance_data = models_and_training.train_model_for_epochs(
        starting_epoch=starting_epoch,
        number_of_epochs_to_train=epochs_to_train,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        prior_performance_data=performance_data,
        head_only=head_only,
        save_confusion_matrix=True
    )

    starting_epoch += epochs_to_train
    models_and_training.save_performance_data(
        performance_data=performance_data, model=model
    )
    models_and_training.save_model(model=model)
    preprocessing.save_train_val_test_indices(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
    )

    print("Session Ended")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Parse all command-line arguments (dataset, model, optimizer)
    # ------------------------------------------------------------
    args = get_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "test":
        run_test(args)
    elif args.command == "plot":
        run_plot(args)
    elif args.command == 'autotrain':
        run_autotrain(args)