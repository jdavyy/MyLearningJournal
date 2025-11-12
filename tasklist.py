
"""
Learning Outcomes: 
-With lists, they obviously cannot be referenced by name so I needed to use a dict to be able to create and reference
multiple lists at a time.
-If we do not use a global variable then every function we call, the dictionary has to be passed as arg every time.

Possible improvements:
-There are many improvements that can be made to this file, one fix would be adding a user input option so instead of running
hard coded tests through main, it would prompt the user for their lists.
-I could save the list as a json file that constantly gets saved to whenever we add or remove tasks. As the source it would keep track
of all the lists individually and could bring up previous tasks despite the past program runtime being closed some time ago. 
-Could add more info to the dictionary and make it so that it can hold data about the item like a due date or a priority
-Optional frontend creation to visualize it all
"""


def create_new_list(tasklist: dict, name: str):
    if name not in tasklist:
        tasklist[name] = []
        print(f"Created list name: {name}")
    else: 
        print("List already exists")

def add_task(tasklist: dict, name: str, task: str):
    if name in tasklist:
        tasklist[name].append(task)
    else: 
        print("List not found in tasklist.")
    
def remove_task(tasklist: dict, name: str, task: str):
    if name in tasklist:
        if task in tasklist[name]:
            tasklist[name].remove(task)
        else:
            print("Task not found in this list")
    else: 
        print("List not found in tasklist.")

def show_list(tasklist: dict, name: str):
    if name not in tasklist: 
        print("List not found in tasklist.")
    else:
        print(f"List: {name}")
        for tasks in tasklist[name]:
            print(f"- {tasks}")


def main():
    tasklist = {}
    create_new_list(tasklist, "fries")
    show_list(tasklist, "fries")
    add_task(tasklist, "fries", "bag them")
    show_list(tasklist, "fries")
    add_task(tasklist, "fries", "cook them")
    show_list(tasklist, "fries")
    remove_task(tasklist, "fries", "cook them")
    show_list(tasklist, "fries")
    create_new_list(tasklist, "burgers")
    show_list(tasklist, "burgers")
    add_task(tasklist, "burgers", "season")
    show_list(tasklist, "burgers")

if __name__ == "__main__":
    main()

