# Contributing Guide 

Thanks for your interest in contributing to the kbtu botskk database!  
Please follow the steps below to submit your information:

## Steps to Contribute
1. **Fork** this repository (click the Fork button in the top right of this page).
2. **Clone** your fork to your local machine:
   ```bash
   git clone https://github.com/<your-username>/<your-forked-repo>.git
   ```
3. **Create a new branch** for your changes:
   ```bash
   git checkout -b add-your-info
   ```
4. **Add your info**:
   Open the dataset.json
   ppend your details in the correct format. Example:
  ```json
  {
  "name": "Your Name",
  "username": "@yourhandle",
  "message": "Hello from me!"
  }
  ```
5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add info for Your Name"
   ```
6.**Push** to your fork:
  ```bash
  git push origin add-your-info
  ```
7.**Open a Pull Request (PR)**
  Go back to your fork on GitHub.
  You’ll see a button to open a PR.
  Provide a short description of what you added.
