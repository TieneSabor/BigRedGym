# Assignment 4

**Submission Deadline:** Friday, October 16th, 11:59PM (midnight)

**Goals:**
- Get familiar with BigRedGym (BRG)
    - Implement a new buffer
    - Implement a new reward
    - Change the observations
    - Tune and iterate

As seen in class (Lecture 16), BRG has two main conceptual places where things take place: the **environment**, implemented in `gym/`, and **runner**, implemented in `learning/`.
The runner describes how the learning process is implemented, including the algorithm, datalogging, data-storage, etc.
You will focus on the environment, which describes the robot and overall the MDP.

Remember, everything important in the environment needs to be _stateful_: it needs to be instantiated during start-up as a torch tensor, so that it can be accessed through setters and getters.

## Goal: control the height

Start by running a full training and try controlling the robot with [`I`,`J`,`K`, `L`, `N`, `M`] for [_forward_, _left-strafe_, _backward_, _right-strafe_, _left-yaw_, _right-yaw_].
(We don't use the more conventional `WASD` because both MuJoCo and VSim, very annoyingly, have reserved some of those keys for specific UI purposes. In fact, MuJoCo has reserved almost every key for _something_, so you'll see some unrelated visuals toggling on and off sometimes if you're using the MuJoCo visualizer. Don't worry about it.)

You task will be to additionally add two controllers (use the up and down arrows) to control the robot base height, and train the robot to stand as tall as possible or crouch low.

### Step 0: make sure you're up to date.

Do a git pull to make sure your `main` is up to date.

### Step 1: sketch out what you'll need.

It is always useful to do some planning first.
On a piece of paper, sketch out what you think you'll need to implement, where you think it'll go, and how you will test that it is doing what you think it should be doing.

### Step 2: branching.

You will soon start a major project. for that it is recommended to not only have a `main` branch that always stays clean and functional, but to branch off a `dev` branch that has _functional_ work-in-progress (WIP).
This means, when you're implementing a new feature or trying something out, branch off of your `dev`, work until you have that feature ready, then merge it into `dev`, similar to how you've so far been merging back into your `main`.
One main advantage of this is that your `main` stays easily sync-able with the upstream `LampLighterLab/BigRedGym`, so when we make updates centrally you can pull it into your repo without merge conflicts.
You can then see if/what you want to pull into your `dev`, if you want to just [cherry-pick](https://www.atlassian.com/git/tutorials/cherry-pick) some commits, etc.

### Step 3: implement it.

Some tips:

- You'll eventually need to add a command to keyboard teleop, which you can do in the file `gym/utils/interfaces/teleop_bindings.py`. We've already taken care of adding the bindings to the up/down arrows, but you need to map this to the appropriate state in your robot environment. Read how other key-board commands are implemented in `TeleopCommands.apply()` and add to it.
- One of the most important implementation details is to get the scaling right. When you implement a new state/buffer, how will you check how it should be normalized?
- Remember to "walk through the code flow" once with the debugger. Don't try to understand every single step in one go, but get an overall sense of things, and come back to things as you go.
- Some useful VSCode commands:
  - `ctrl`+`shift`+`P` (`cmd`+`shift`+`P` on macOS; generally, `ctrl` is replaced with `cmd` on macOS) will bring up the Command Palette, from where you can find any action. You should get very comfortable doing things from here instead of clicking through drop-downs.
  - From the Command Palette, you can apply `Fold All` (or `Unfold All`) to fold all indented code, which is helpful for getting an overview of long code-files.
  - `ctrl`+`P` lets you go to any file, with lazy-search (you can type parts of the filename in any order, and it will show you matches). Very useful for navigating around different files.
  - `ctrl`+`tab` and `ctrl`+`shift`+`tab` (this one is also `ctrl` on macOS) lets you cycle forward and backward through tabs (just like in a browser). Side-note, I highly recommend remapping your caps-lock key to a second `ctrl` key; the `ctrl` key is super useful in the terminal and everywhere, caps-lock is useless; but it's in a much more ergonomic position on the keyboard.
  - The command 'go to definition' with a variable/function selected will take you to its definition, in whichever file it should be. You can find it by right-clicking a variable/function or look up the hotkey (I have it mapped to `cmd`+`k`; `cmd`+`g`, but I don't think that's the default).
  - `ctrl`+`D` will select the word your cursor is at; press it again to also select the next instance of the word, ad nauseum; you now have multiple-cursors and can replace many things at once. You can also instantly select all instances, or manually place cursors with `alt`+`click` (`opt`+`click` on mac).
- You are encouraged to talk to each other and share notes, especially on Ed.

### Optional: Weights & Biases

BRG has Weights & Biases integration... which has recently been renamed 'CoreWeave Forge'. I will continue to refer to it as WandB, which is also how it is still referred to in the code.
WandB is a cloud-based progress tracker for machine-learning runs, which makes it easier to have an overview of reward curves, performance, etc. of lots of training runs.
It's very helpful to be able to compare progress, keep track of things, etc.
It's still free for academic use, so I encourage you to create your account and try it.
If you're familiar with tensorboard, this has a similar purpose.

### Optional: Code Review

You will have an assigned partner; however, you will have no deliverable associated with code-reviewing.
You are encouraged to open a PR (to your own repo) and connect with your partner to trade code-reviews.

### Deliverables

Submit a single screen-recording of your trained robot positioning its base as low as possible, and as high as possible, and cycling between these twice (low-high-low-high).
You should be able to take this in one shot, since you will have implemented the keyboard commands to move it up and down.
It should be able to transition to and from both of these stably.

In addition, submit a PDF with a time-series plot showing how well your policy actually does, quantitatively.