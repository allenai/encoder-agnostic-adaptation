import os
import argparse
import re

ENCODER_AGNOSTIC_PAD = "SHALL"
TRIM_END_IF_FOUND = [' --', ' cut into']
TRIM_START_IF_FOUND = ['-']
REGEX_TO_REPLACE = """(\(?(\d+\s)*\d+([\/-]\d+)?\s*(?!(minutes))(scoops|-inch|inch|cl|ts|tb|oz|sm|ea|sl|lb|ds|qt|cn|lg|md|pk|pt|to|in.|pn|dr|ct|bn|ea|t|ts|T|fl|ga|ml|cb|dl|mg|cg|dg|kg|l|g|"|c|t|x|)*\s*\)?;?\s+)*([;,\s]*(sized|finely|chopped|minced|pkg|\(optional\))\s*)*"""

def merge_recipes(opt):
    clean_folder = os.path.join(opt.dataset, "clean")
    titles, ingredients, recipe_bodies = [], [], []
    num_of_recipes, num_skipped = 0, 0
    for file in os.listdir(clean_folder):
        path = os.path.join(clean_folder, file)
        with open(path, 'r') as f:
            lines = f.readlines()

        lines = remove_blanks(lines)
        recipes = split_list_by_value(lines, "END RECIPE\n")
        num_of_recipes += len(recipes)

        for recipe in recipes:
            ingredient, skipped = prepare_ingredients(recipe[3])
            if skipped or len(ingredient) == 0:
                num_skipped += 1
                continue
            ingredients.append(ingredient)
            titles.append(prepare_titles(recipe[0]))
            recipe_bodies.append(prepare_recipe_body(recipe[4:]).strip())

    print(f"Skipped {num_skipped} recipes out of {num_of_recipes}.")

    write_list_to_file(titles, f"{opt.output}.txt.src")
    write_list_to_file(recipe_bodies, f"{opt.output}.txt.tgt")
    write_list_to_file(ingredients, f"{opt.output}.txt.agenda")

def prepare_ingredients(ingredients):
    ingredients = clean_prefix(ingredients, "ingredients: ")
    ingredient_list = ingredients.split('\t')
    ret_ingredients = []
    for ingredient in ingredient_list:
        ingredient = re.sub(REGEX_TO_REPLACE, '', ingredient)
        ingredient = ingredient.replace(",", "")
        ingredient = ingredient.replace(";", "")
        ingredient = ingredient.replace("^", "")
        ingredient = ingredient.replace("*", "")
        ingredient = ingredient.replace(".", "")
        ingredient = ingredient.replace("&", "")
        ingredient = ingredient.replace(":", "")
        ingredient = ingredient.replace("<", "")
        ingredient = ingredient.replace(">", "")
        ingredient = ingredient.replace("~", "")
        ingredient = ingredient.replace("=", "")
        ingredient = ingredient.replace("+", "")
        ingredient = ingredient.replace("(", "")
        ingredient = ingredient.replace(")", "")
        ingredient = ingredient.replace(" - ", " ")
        ingredient = ingredient.replace(" or ", " ")
        ingredient = ingredient.replace(" or ", " ")
        ingredient = ingredient.replace("-lrb-", "")
        ingredient = ingredient.replace("-LRB-", "")
        ingredient = ingredient.replace("-rrb-", "")
        ingredient = ingredient.replace("-RRB-", "")
        ingredient = ingredient.replace("a-1? ", "")
        for pattern in TRIM_END_IF_FOUND:
            if pattern in ingredient:
                pattern_index = ingredient.index(pattern)
                ingredient = ingredient[:pattern_index]
        for pattern in TRIM_START_IF_FOUND:
            if pattern in ingredient:
                pattern_index = ingredient.index(pattern)
                ingredient = ingredient[pattern_index+1:]
        ingredient = re.sub(" +", " ", ingredient)
        ret_ingredients.append(ingredient)

    return f" {ENCODER_AGNOSTIC_PAD} ".join(ret_ingredients).strip(), False

def prepare_ingredients_using_body(body):
    ingredients = set()
    it = iter(body)
    for recipe, annotation in zip(it, it):
        recipe_list = recipe.split()
        annotation_list = annotation.split()
        if len(recipe_list) != len(annotation_list):
            return None, True
        for i, ann in enumerate(annotation_list):
            if ann.startswith('1_'):
                ingredients.add(recipe_list[i].replace('_', ' '))

    return f' {ENCODER_AGNOSTIC_PAD} '.join(ingredients), False

def prepare_titles(title):
    return clean_prefix(title, "title: ").strip()

def prepare_recipe_body(body):
    body_str = ''
    for recipe in body:
        body_str += recipe.replace('_', ' ').replace('\n', ' ')

    return body_str

def clean_prefix(val, prefix):
    assert val.lower().startswith(prefix), \
        f"{val} should have started with {prefix}"
    return val[len(prefix):]

def write_list_to_file(list, path):
    with open(path, 'w') as f:
        for v in list:
            f.write(f"{v}\n")

def remove_blanks(list):
    return [v for v in list if v != '\n']

def split_list_by_value(list, value):
    def _remove_last_if_end_with_value(ret):
        if ret[-1] == []:
            return ret[:-1]
        return ret

    ret = [[]]
    for v in list:
        if v == value:
            ret.append([])
        else:
            ret[-1].append(v)

    return _remove_last_if_end_with_value(ret)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-dataset', type=str, required=True)
    parser.add_argument('--output', '-output', type=str, required=True)

    opt = parser.parse_args()
    merge_recipes(opt)