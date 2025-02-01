---
title: How to programmatically post your personal blogs to Dev.to, Hashnode, and Medium with Github actions.
---
Welcome to the launch of our exciting new series, *WTF is Artificial Intelligence?*  
In this 30-day journey, we will delve into the world of Machine Learning, Deep Learning, and Generative AI, breaking down complex concepts into clear, practical explanations. Whether you’re a beginner or just looking for a refresher, this series will guide you step-by-step from zero to hero.

Let’s kick things off with *Day 0*, where we’ll explore the fundamentals of Machine Learning: what it is, why it matters, and how it’s transforming the world around us. No jargon, no buzzwords—just straightforward learning for everyone.

---

## Why This Tutorial?

By the end of this tutorial, you’ll:  
- ✅ Understand the foundational concepts of ML in simple, clear terms.  
- ✅ Explore real-world applications of ML and why it’s so important.  
- ✅ Learn about the three main types of ML with practical examples and code.  

*Let’s roll up our sleeves and dive right in!*

---

## What is Machine Learning?

*Machine Learning (ML)* is all about teaching computers to learn from data and make decisions or predictions without being explicitly programmed. Think of it like showing a child pictures of cats and dogs. Over time, the child learns to recognize which is which. Similarly, ML enables machines to identify patterns and make intelligent decisions based on data.

But why does this matter? Because it allows us to automate tasks that are too complex for traditional programming. From diagnosing diseases to recommending movies, ML is everywhere.

---

## Why Should You Care About ML?

Machine Learning powers the technologies you use every day. When Netflix recommends your next binge-worthy show or your smartphone auto-corrects a typo, that’s ML in action. Self-driving cars, voice assistants, and even those ads that seem to know exactly what you want? All thanks to Machine Learning.

It’s not just about convenience. ML is revolutionizing industries, helping us solve problems faster and more efficiently than ever before.

---

## Types of Machine Learning

ML is typically categorized into three types. Let’s break them down:

### 1. Supervised Learning



![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/dshtctnle652jm9ytd62.png)

Supervised learning is like learning with a teacher. The machine is trained using labeled data, where each input has a corresponding output.  

*Example:*  
Predicting house prices using features like size, bedrooms, and neighborhood.  

*Code Example:*  
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample Data: Features (size, bedrooms) and Labels (price)
X = [[1200, 3], [1500, 4], [800, 2], [2000, 5]]  # Features
y = [300000, 400000, 200000, 500000]  # Labels

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

## Make Predictions

predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
```

### 2. Unsupervised Learning
Unsupervised learning works without labeled data. The machine explores the data to find patterns or groupings on its own.

Example:
Grouping customers based on purchase habits.

```python
from sklearn.cluster import KMeans
import numpy as np

# Sample Data
data = np.array([[1, 2], [2, 3], [10, 11], [11, 12]])

# Create and Train the Model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)

# Output the Cluster Centers and Labels
print("Cluster Centers:", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)
```

### 3. Reinforcement Learning
Reinforcement learning is inspired by how humans learn through rewards and punishments. The machine takes actions in an environment and learns from the outcomes.

Example:
A robot learning to walk through trial and error.

Code Example (Conceptual):
```python
import gym

# Create Environment
env = gym.make("CartPole-v1")

# Initialize Variables
state = env.reset()
done = False

# Simple Loop to Take Random Actions
while not done:
    action = env.action_space.sample()  # Take Random Action
    state, reward, done, info = env.step(action)
    env.render()  # Visualize the Environment

env.close()

```
## Why Machine Learning Matters in the Real World

Machine Learning is transforming industries and solving problems in ways we couldn’t imagine a decade ago. Here are a few examples:

- 🏥 *Healthcare*: Early disease detection, personalized treatment plans, and drug discovery.  
- 💰 *Finance*: Fraud detection, automated trading, and personalized financial advice.  
- 🛍 *Retail*: Predicting inventory needs, optimizing supply chains, and personalized shopping experiences.  
- 🎮 *Entertainment*: Curated playlists on Spotify and personalized recommendations on Netflix.  

---

## Key Takeaways:
- Machine Learning enables machines to learn from data and make decisions.  
- It can be categorized into *supervised, **unsupervised, and **reinforcement learning*.  
- ML is transforming industries, making it an essential technology for the future.  

---

## Resources to Explore:
Want to dive deeper? Check out these resources:  
- [Andrew Ng’s Machine Learning Course (Coursera)](https://www.coursera.org/learn/machine-learning)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)  
- [Google’s Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)  

---

## Thanks for Joining the Journey!

Thanks for joining me on this learning adventure! This is just *Day 2* of our series, and we’re already uncovering key Machine Learning concepts. Over the next 30 days, we’ll simplify complex topics, dive into hands-on examples, and share cheat sheets to make your ML journey both fun and rewarding.  

---

### What’s Coming Up Next?
Later this week, we’ll cover exciting topics to build your foundational knowledge in Machine Learning:  

1. *Day 2*: Supervised Learning Made Simple → Explore the basics of supervised models with practical examples.  
2. *Day 3*: Unsupervised Learning Made Easy → Explore clustering, dimensionality reduction, and how to detect anomalies.  
3. *Day 4*: Regression in ML → Learn how linear regression works for predicting continuous values.  
4. *Day 5*: Classification Models → Dive into Logistic Regression and Decision Trees for real-world use cases.  
5. *Day 6*: Introduction to Reinforcement Learning → Understand agents, environments, and rewards—and see it in action.  
6. *Day 7*: Data Preprocessing in ML → Discover how to clean, scale, and encode your data for better results.  

---

## Let’s Connect and Build Together

Make sure to follow me on [X (formerly Twitter)](https://twitter.com/) and turn on notifications to stay updated on all the latest tutorials. Together, we’ll make Machine Learning accessible and fun for everyone.  

Here’s how we can collaborate:  
- Open to DevRel partnerships to help brands grow through educational content.  
- Have an AI MVP idea or need consultancy services for AI-based applications and research projects? Let’s make it happen!  

📧 *Drop a mail at*: [thenomadevel@gmail.com](mailto:thenomadevel@gmail.com)
