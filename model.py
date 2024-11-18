{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6bcf357d-d5a6-4bd0-bad4-52fbdd1d1dba",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f990be89-493a-4bfc-a2a0-4ffdb31e9949",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9989244007375537\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd \n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import pickle\n",
    "\n",
    "# Load the dataset\n",
    "df = pd.read_csv('dataset.csv')\n",
    "\n",
    "# Split the data into features (X) and target (y)\n",
    "X = df.iloc[:, :-1]\n",
    "y = df.iloc[:, -1]\n",
    "\n",
    "# Split the data into training and test sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)\n",
    "\n",
    "# Define the ColumnTransformer\n",
    "trf = ColumnTransformer([\n",
    "    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])\n",
    "], remainder='passthrough')\n",
    "\n",
    "# Define the Pipeline\n",
    "ra_pipe = Pipeline([\n",
    "    ('step1', trf),\n",
    "    ('step2', RandomForestClassifier(random_state=100))  # Setting random_state for reproducibility\n",
    "])\n",
    "\n",
    "# Train the pipeline on the training data\n",
    "ra_pipe.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions on the test data\n",
    "y_pred = ra_pipe.predict(X_test)\n",
    "\n",
    "# Evaluate the model's accuracy\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "\n",
    "# Save the trained pipeline to a file\n",
    "with open('ra_pipe.pkl', 'wb') as file:\n",
    "    pickle.dump(ra_pipe, file)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c0ee035-f798-4a96-af67-5d1f00cd7b93",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
