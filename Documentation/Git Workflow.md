# Git Workflow

1. Create Issue

    ![Create Issue](Images/Untitled.png)

2. Create new branch from dev named like issue

    You can create a new branch directly on Git, or you can create a new branch the following:

    ```bash
    git checkout dev
    git pull
    git branch issue
    git checkout issue
    ```

3. Commit your changes and push to issue branch
4. Checkout to dev branch

    ```bash
    git checkout dev
    ```

5. Pull newest dev branch

    ```bash
    git pull
    ```

6. Checkout to issue branch

    ```bash
    git checkout issue
    ```

7. Add new changes from dev branch to feature branch

    ```bash
    git merge dev
    ```

    ![git merge main](Images/Untitled%201.png)

8. Commit and push all changes to origin

    ```bash
    git commit -m "Merged changes on dev to issue branch"
    git push
    ```

9. Create pull request to merge issue branch to dev branch.
Choose the branch you want to merge on the right (here kmeans) and the dev branch on the left.
    ![Create Pull request](Images/pullRequest3.png)
