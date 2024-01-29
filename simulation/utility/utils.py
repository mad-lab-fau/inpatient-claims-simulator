import numpy as np


def get_gender_list() -> list:
    """Get a list of all possible genders.

    Returns:
        a list: ["weiblich", "männlich", "Insgesamt"]
    """
    return ["weiblich", "männlich", "Insgesamt"]


def get_gender_group(gender: str) -> str:
    """Get the group according to a gender.

    Args:
        gender (str): the gender initial (e.g. "M" or "F" or anything else)

    Returns:
        a String to get the group name in the statistics files
    """
    if gender == "M":
        return "männlich"
    elif gender == "F":
        return "weiblich"
    else:
        return "Insgesamt"


def get_age_list() -> list:
    """Get a list of all possible age groups.

    Returns:
        a list of every possible age group/cluster.
    """
    return [
        "unter 1 Jahr",
        "1 bis unter 5 Jahre",
        "5 bis unter 10 Jahre",
        "10 bis unter 15 Jahre",
        "15 bis unter 18 Jahre",
        "18 bis unter 20 Jahre",
        "20 bis unter 25 Jahre",
        "25 bis unter 30 Jahre",
        "30 bis unter 35 Jahre",
        "35 bis unter 40 Jahre",
        "40 bis unter 45 Jahre",
        "45 bis unter 50 Jahre",
        "50 bis unter 55 Jahre",
        "55 bis unter 60 Jahre",
        "60 bis unter 65 Jahre",
        "65 bis unter 70 Jahre",
        "70 bis unter 75 Jahre",
        "75 bis unter 80 Jahre",
        "80 bis unter 85 Jahre",
        "85 bis unter 90 Jahre",
        "90 bis unter 95 Jahre",
        "95 Jahre und mehr",
    ]


def get_age_group(age: int) -> str:
    """Get the name of the age group according to an age.

    Args:
        age (int): the age of the patient

    Returns:
        a String of the group name
    """
    if age < 1:
        return "unter 1 Jahr"
    elif age >= 1 and age < 5:
        return "1 bis unter 5 Jahre"
    elif age >= 5 and age < 10:
        return "5 bis unter 10 Jahre"
    elif age >= 10 and age < 15:
        return "10 bis unter 15 Jahre"
    elif age >= 15 and age < 18:
        return "15 bis unter 18 Jahre"
    elif age >= 18 and age < 20:
        return "18 bis unter 20 Jahre"
    elif age >= 20 and age < 25:
        return "20 bis unter 25 Jahre"
    elif age >= 25 and age < 30:
        return "25 bis unter 30 Jahre"
    elif age >= 30 and age < 35:
        return "30 bis unter 35 Jahre"
    elif age >= 35 and age < 40:
        return "35 bis unter 40 Jahre"
    elif age >= 40 and age < 45:
        return "40 bis unter 45 Jahre"
    elif age >= 45 and age < 50:
        return "45 bis unter 50 Jahre"
    elif age >= 50 and age < 55:
        return "50 bis unter 55 Jahre"
    elif age >= 55 and age < 60:
        return "55 bis unter 60 Jahre"
    elif age >= 60 and age < 65:
        return "60 bis unter 65 Jahre"
    elif age >= 65 and age < 70:
        return "65 bis unter 70 Jahre"
    elif age >= 70 and age < 75:
        return "70 bis unter 75 Jahre"
    elif age >= 75 and age < 80:
        return "75 bis unter 80 Jahre"
    elif age >= 80 and age < 85:
        return "80 bis unter 85 Jahre"
    elif age >= 85 and age < 90:
        return "85 bis unter 90 Jahre"
    elif age >= 90 and age < 95:
        return "90 bis unter 95 Jahre"
    else:
        return "95 Jahre und mehr"


def flatten(arr: list) -> list:
    """Unpack all sub-liststo get one list of singular values.

    Args:
        arr (list): a list of values that may contain sub-lists

    Returns:
        a list of of singular values (without sub-lists)
    """
    flat_arr = []
    for item in arr:
        if isinstance(item, list):
            flat_arr.extend(flatten(item))
        else:
            flat_arr.append(item)
    return flat_arr


def poisson_greater_than_zero(lmbda: float, random_state: int = None) -> int:
    """Getting a value with a Poisson-distribution that is greater than 0

    Args:
        lmbda (float): the lambda-value for np.random.poisson
        random_state (int, optional): random state for reproducibility. Defaults to None.

    Returns:
        a positive `int` value
    """

    if random_state is not None:
        np.random.seed(random_state)
    x = 0
    while x == 0:
        x = np.random.poisson(lmbda)

    return x


def strip_nA(list: list) -> list:
    """Get a version of a list of values without 'n/A'.

    Args:
        list (list): a list of Strings (e.g. list of OPS)

    Returns:
        a `list` with the same values but stripped of 'n/A'
    """
    return [i for i in list if i != "n/A"]
