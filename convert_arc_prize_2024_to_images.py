from src.datasets.arc_prize_2024 import json_file_read
from src.datasets.arc_prize_2024 import challenges_to_image_files


def convert_arc_prize_2024_to_images(skip_if_working_dir_exists=True):
    training_challenges = json_file_read('datasets/arc-prize-2024/json/arc-agi_training_challenges.json')
    training_solutions = json_file_read('datasets/arc-prize-2024/json/arc-agi_training_solutions.json')

    evaluation_challenges = json_file_read('datasets/arc-prize-2024/json/arc-agi_evaluation_challenges.json')
    evaluation_solutions = json_file_read('datasets/arc-prize-2024/json/arc-agi_evaluation_solutions.json')

    challenges_to_image_files(
        'datasets/arc-prize-2024/images/training',
        training_challenges,
        training_solutions,
        image_variations=10,
        skip_if_working_dir_exists=skip_if_working_dir_exists
    )

    challenges_to_image_files(
        'datasets/arc-prize-2024/images/evaluation',
        evaluation_challenges,
        evaluation_solutions,
        image_variations=False,
        skip_if_working_dir_exists=skip_if_working_dir_exists
    )


if __name__ == '__main__':
    convert_arc_prize_2024_to_images(skip_if_working_dir_exists=False)
