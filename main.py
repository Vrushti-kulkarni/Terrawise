import pandas as pd
from faker import Faker
import random

faker = Faker()

# Static user profile data
locations = ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata', 'pune']
diets = ['veg', 'non_veg', 'vegan']
transports = ['car', 'bike', 'public_transport', 'walking', 'rideshare']
goals = ['zero_waste', 'reduce_emissions', 'plastic_free', 'energy_efficient', 'water_conservation']
interests = ['gardening', 'cooking', 'technology', 'outdoors', 'fashion', 'finance', 'education']
dwellings = ['apartment', 'house', 'shared']

# Contextual dimensions
times = ['morning', 'afternoon', 'evening', 'night', 'weekend']
ctx_locations = ['home', 'commute', 'work']
weather = ['sunny', 'rainy', 'hot', 'cold']
moods = ['happy', 'tired', 'motivated', 'bored']
companions = ['alone', 'with_friends', 'with_family']
devices = ['phone', 'laptop', 'tablet']

# Mapping user goals to relevant actions
goal_action_map = {
    'energy_efficient': ['turn_off_appliances', 'set_ac_to_24_degrees', 'fix_leaky_taps'],
    'zero_waste': ['carry_reusable_bag', 'compost_kitchen_waste', 'use_refillable_bottles'],
    'plastic_free': ['use_refillable_bottles', 'carry_reusable_bag', 'avoid_single_use_plastic'],
    'reduce_emissions': ['use_public_transport', 'buy_local_produce', 'take_shorter_showers'],
    'water_conservation': ['fix_leaky_taps', 'take_shorter_showers', 'set_ac_to_24_degrees']
}

# All recommended actions
all_actions = [
    "carry_reusable_bag",
    "use_public_transport",
    "turn_off_appliances",
    "take_shorter_showers",
    "avoid_single_use_plastic",
    "compost_kitchen_waste",
    "buy_local_produce",
    "use_refillable_bottles",
    "set_ac_to_24_degrees",
    "fix_leaky_taps"
]

# Generate user profiles
users = []
for i in range(1000):
    user = {
        'user_id': f'user_{i}',
        'location': random.choice(locations),
        'diet': random.choice(diets),
        'transport': random.choice(transports),
        'goal': random.choice(goals),
        'age_group': random.choice(['18-24', '25-34', '35-44', '45-54', '55+']),
        'interest': random.choice(interests),
        'dwelling': random.choice(dwellings)
    }
    users.append(user)

users_df = pd.DataFrame(users)

interactions = []

for user in users:
    for _ in range(random.randint(3, 7)):
        ctx_loc = random.choice(ctx_locations)
        device = random.choice(['phone', 'laptop', 'tablet']) if ctx_loc != 'commute' else random.choice(['phone', 'tablet'])
        companion = random.choice(['alone', 'with_family']) if ctx_loc == 'home' else random.choice(['alone', 'with_friends'])

        weather_now = random.choice(weather)
        mood_now = random.choice(moods)

        # Get user's sustainability goal
        user_goal = user['goal']

        # Smart filtering based on weather and mood
        if weather_now == 'rainy':
            avoid_actions = ['compost_kitchen_waste', 'buy_local_produce']
        elif weather_now == 'hot':
            avoid_actions = ['compost_kitchen_waste']
        else:
            avoid_actions = []

        # Action selection with a weighted approach
        r = random.random()
        if r < 0.6:  # 60% chance to recommend goal-aligned action
            # Get primary actions aligned with the user's goal
            potential_actions = goal_action_map[user_goal].copy()
            # Filter out actions to avoid based on weather
            filtered_actions = [a for a in potential_actions if a not in avoid_actions]
            if not filtered_actions:  # If all goal actions filtered out
                filtered_actions = potential_actions  # Ignore weather constraints
        elif r < 0.85:  # 25% chance to recommend secondary related actions
            # Get actions from other related goals
            if user_goal == 'energy_efficient' or user_goal == 'water_conservation':
                related_goals = ['energy_efficient', 'water_conservation']
            else:
                related_goals = ['zero_waste', 'plastic_free', 'reduce_emissions']
                
            related_goals = [g for g in related_goals if g != user_goal]
            potential_actions = []
            for g in related_goals:
                potential_actions.extend(goal_action_map[g])
            filtered_actions = [a for a in potential_actions if a not in avoid_actions]
        else:  # 15% chance to recommend any action (for exploration/diversity)
            filtered_actions = [a for a in all_actions if a not in avoid_actions]
            
        if not filtered_actions:
            filtered_actions = all_actions

        # Recommendation decision
        recommended_action = random.choice(filtered_actions)

        # Determine if the action is aligned with the user's goal
        is_goal_aligned = recommended_action in goal_action_map[user_goal]
        
        # More complex acceptance logic with various factors
        # Base acceptance probability
        if is_goal_aligned:
            # Goal-aligned actions have higher base probability but still with variation
            base_prob = random.uniform(0.5, 0.8)
        else:
            # Non-aligned actions have lower base probability but could still be accepted
            base_prob = random.uniform(0.2, 0.5)
            
        # Context modifiers
        context_modifier = 0.0
        
        # Mood impact
        if mood_now == 'motivated':
            context_modifier += random.uniform(0.05, 0.15)
        elif mood_now == 'tired':
            context_modifier -= random.uniform(0.05, 0.15)
        elif mood_now == 'bored':
            # Some actions might be more appealing when bored
            if recommended_action in ['buy_local_produce', 'compost_kitchen_waste']:
                context_modifier += random.uniform(0.05, 0.15)  # These could be engaging activities
            else:
                context_modifier -= random.uniform(0.02, 0.1)
                
        # Weather impact
        if weather_now == 'rainy' and recommended_action in ['use_public_transport', 'buy_local_produce']:
            context_modifier -= random.uniform(0.1, 0.2)
        elif weather_now == 'hot' and recommended_action in ['take_shorter_showers', 'set_ac_to_24_degrees']:
            context_modifier += random.uniform(0.05, 0.15)
            
        # Location impact
        if ctx_loc == 'home' and recommended_action in ['turn_off_appliances', 'fix_leaky_taps', 'set_ac_to_24_degrees']:
            context_modifier += random.uniform(0.05, 0.15)
        elif ctx_loc == 'commute' and recommended_action in ['use_public_transport', 'carry_reusable_bag']:
            context_modifier += random.uniform(0.05, 0.15)
        elif ctx_loc == 'work' and recommended_action in ['use_refillable_bottles', 'avoid_single_use_plastic']:
            context_modifier += random.uniform(0.05, 0.15)
            
        # Companion impact
        if companion == 'with_friends' and recommended_action in ['use_public_transport', 'buy_local_produce']:
            context_modifier += random.uniform(0.05, 0.1)  # Social influence
            
        # Final acceptance probability (bounded between 0.1 and 0.9 to ensure variety)
        acceptance_prob = max(0.1, min(0.9, base_prob + context_modifier))
        
        # Determine acceptance
        accepted = 1 if random.random() < acceptance_prob else 0

        interaction = {
            'user_id': user['user_id'],
            'time': random.choice(times),
            'context_location': ctx_loc,
            'weather': weather_now,
            'mood': mood_now,
            'companion': companion,
            'device': device,
            'recommended_action': recommended_action,
            'accepted': accepted,
            'goal_aligned': is_goal_aligned  # Track if recommendation aligns with goal (useful for analysis)
        }

        interactions.append(interaction)

interactions_df = pd.DataFrame(interactions)

# Merging users' data into interactions dataframe
interactions_df = interactions_df.merge(users_df[['user_id', 'goal', 'age_group', 'location', 'diet', 'transport']], on='user_id', how='left')

print("Sample of interactions:")
print(interactions_df.head(10))
print("\nDataframe columns:")
print(interactions_df.columns)

# Verify distribution of recommendations and acceptance rates
print("\nDistribution of recommended actions:")
action_counts = interactions_df['recommended_action'].value_counts(normalize=True)
print(action_counts)

print("\nAcceptance rate by goal alignment:")
alignment_acceptance = interactions_df.groupby('goal_aligned')['accepted'].mean()
print(alignment_acceptance)

print("\nAcceptance rate by goal and recommended action:")
goal_action_acceptance = interactions_df.groupby(['goal', 'recommended_action'])['accepted'].agg(['count', 'mean']).reset_index()
print(goal_action_acceptance.sort_values(['goal', 'mean'], ascending=[True, False]))

# Check if there's still a strong relationship between goal and recommended action
print("\nGoal vs. recommended action confusion matrix:")
goal_action_matrix = pd.crosstab(interactions_df['goal'], interactions_df['recommended_action'], normalize='index')
print(goal_action_matrix)

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set plot style
# sns.set(style="whitegrid")

# # Plot 1: Distribution of Locations in Users DataFrame
# plt.figure(figsize=(10, 6))
# sns.countplot(data=users_df, x='location', palette='viridis')
# plt.title('Distribution of Users by Location')
# plt.xlabel('Location')
# plt.ylabel('Number of Users')
# plt.xticks(rotation=45)
# plt.show()

# # Plot 2: Diet Preferences in Users DataFrame
# plt.figure(figsize=(10, 6))
# sns.countplot(data=users_df, x='diet', palette='coolwarm')
# plt.title('Diet Preferences of Users')
# plt.xlabel('Diet Type')
# plt.ylabel('Number of Users')
# plt.show()

# # Plot 3: Transport Mode Preferences in Users DataFrame
# plt.figure(figsize=(10, 6))
# sns.countplot(data=users_df, x='transport', palette='magma')
# plt.title('Transport Mode Preferences of Users')
# plt.xlabel('Transport Mode')
# plt.ylabel('Number of Users')
# plt.show()

# # Plot 4: Accepted vs Rejected Actions in Interactions DataFrame
# plt.figure(figsize=(10, 6))
# sns.countplot(data=interactions_df, x='accepted', palette='coolwarm')
# plt.title('Accepted vs Rejected Actions')
# plt.xlabel('Accepted (1) / Rejected (0)')
# plt.ylabel('Count of Interactions')
# plt.show()

# # Plot 5: Acceptance by Weather Condition in Interactions DataFrame
# plt.figure(figsize=(10, 6))
# sns.countplot(data=interactions_df, x='weather', hue='accepted', palette='Set2')
# plt.title('Acceptance of Actions by Weather')
# plt.xlabel('Weather')
# plt.ylabel('Count of Interactions')
# plt.show()

# # Plot 6: Acceptance by Time of Day in Interactions DataFrame
# plt.figure(figsize=(10, 6))
# sns.countplot(data=interactions_df, x='time', hue='accepted', palette='Set1')
# plt.title('Acceptance of Actions by Time of Day')
# plt.xlabel('Time of Day')
# plt.ylabel('Count of Interactions')
# plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Assuming interactions_df and users_df are already loaded and merged.

# Step 1.1: Handle missing values
# interactions_df = interactions_df.fillna(method='ffill')  # Forward fill if missing values exist

# Step 1.2: Define categorical columns for OneHotEncoding and LabelEncoding
categorical_columns_onehot = ['time', 'goal', 'weather', 'mood', 'context_location', 'companion', 'device']
categorical_columns_label = ['age_group', 'location', 'diet', 'transport','recommended_action']

# Step 1.3: Create a ColumnTransformer with separate pipelines for OneHot and Label Encoding
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()

print("columns are ", interactions_df.columns)

# Check if 'age_group' is in the DataFrame
if 'age_group' in interactions_df.columns:
    le1 = LabelEncoder()
    interactions_df['age_group_encoded'] = le1.fit_transform(interactions_df['age_group'])
else:
    print("❌ 'age_group' column is missing in interactions_df.")

interactions_df['location_encoded'] = le2.fit_transform(interactions_df['location'])
interactions_df['diet_encoded'] = le3.fit_transform(interactions_df['diet'])
interactions_df['transport_encoded'] = le4.fit_transform(interactions_df['transport'])
interactions_df['recommended_action_encoded'] = le5.fit_transform(interactions_df['recommended_action'])


from sklearn.preprocessing import OneHotEncoder

# OneHot columns
categorical_columns_onehot = ['time', 'goal', 'weather', 'mood', 'context_location', 'companion', 'device']

# Create individual encoders
ohe1 = OneHotEncoder()
ohe2 = OneHotEncoder()
ohe3 = OneHotEncoder()
ohe4 = OneHotEncoder()
ohe5 = OneHotEncoder()
ohe6 = OneHotEncoder()
ohe7 = OneHotEncoder()

# Force OneHotEncoder to return dense arrays
encoded_time = pd.DataFrame(ohe1.fit_transform(interactions_df[['time']]).toarray(), 
                            columns=ohe1.get_feature_names_out(['time']))
encoded_goal = pd.DataFrame(ohe2.fit_transform(interactions_df[['goal']]).toarray(), 
                            columns=ohe2.get_feature_names_out(['goal']))
encoded_weather = pd.DataFrame(ohe3.fit_transform(interactions_df[['weather']]).toarray(), 
                               columns=ohe3.get_feature_names_out(['weather']))
encoded_mood = pd.DataFrame(ohe4.fit_transform(interactions_df[['mood']]).toarray(), 
                            columns=ohe4.get_feature_names_out(['mood']))
encoded_context_location = pd.DataFrame(ohe5.fit_transform(interactions_df[['context_location']]).toarray(), 
                                        columns=ohe5.get_feature_names_out(['context_location']))
encoded_companion = pd.DataFrame(ohe6.fit_transform(interactions_df[['companion']]).toarray(), 
                                 columns=ohe6.get_feature_names_out(['companion']))
encoded_device = pd.DataFrame(ohe7.fit_transform(interactions_df[['device']]).toarray(), 
                              columns=ohe7.get_feature_names_out(['device']))


# Combine all encoded columns into one DataFrame
encoded_df = pd.concat([
    encoded_time,
    encoded_goal,
    encoded_weather,
    encoded_mood,
    encoded_context_location,
    encoded_companion,
    encoded_device,
    interactions_df[['age_group_encoded', 'location_encoded', 'diet_encoded', 'transport_encoded','recommended_action_encoded']]
], axis=1)

print(encoded_df.head(10))

print(interactions_df.head())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Step 1: Define target and features
X = encoded_df.copy()
y = interactions_df['accepted']  # Binary target from original DataFrame

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)



# Step 4: Predict
y_pred = rf.predict(X_test)

# Step 5: Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(rf, 'random_forest_model.pkl')

print("✅ Model saved as 'random_forest_model.pkl'")

# Define your actions
actions = ['turn_off_appliances', 'fix_leaky_taps', 'take_shorter_showers',
       'carry_reusable_bag', 'set_ac_to_24_degrees',
       'use_public_transport', 'avoid_single_use_plastic',
       'compost_kitchen_waste', 'buy_local_produce',
       'use_refillable_bottles']  # List all your actions

# Function to get the predicted acceptance for each action
def get_best_action(user_context, actions, model):
    predicted_acceptances = []

    # Iterate through each action and predict acceptance
    for action in actions:    
        encoded_action = le5.transform([action])[0]  # Convert action string to encoded int

        user_context['recommended_action_encoded'] = encoded_action
        
        # Convert user context into DataFrame (same format as your training data)
        users_df = pd.DataFrame([user_context])  # Your user context should be formatted as a DataFrame

        users_df = users_df[encoded_df.columns]

        
        # Predict the probability of acceptance for this action (1 indicates acceptance)
        acceptance_score = model.predict_proba(users_df)[:, 1]  # We use the probability of 'accepted' (class 1)
        
        # Store the action and its predicted acceptance score
        predicted_acceptances.append((action, acceptance_score[0]))

    # Sort the actions by predicted acceptance score (highest first)
    recommended_action = sorted(predicted_acceptances, key=lambda x: x[1], reverse=True)[0]
    
    return recommended_action

# Example user context (this would be provided in real use cases)
# user_context = {
#     # Time (one-hot, only one should be 1)
#     'time_morning': 1,
#     'time_afternoon': 0,
#     'time_evening': 0,
#     'time_night': 0,
#     'time_weekend': 0,

#     # Goal (one-hot)
#     'goal_energy_efficient': 0,
#     'goal_plastic_free': 1,
#     'goal_reduce_emissions': 0,
#     'goal_water_conservation': 0,
#     'goal_zero_waste': 0,

#     # Weather (one-hot)
#     'weather_sunny': 1,
#     'weather_cold': 0,
#     'weather_hot': 0,
#     'weather_rainy': 0,

#     # Mood (one-hot)
#     'mood_happy': 0,
#     'mood_motivated': 1,
#     'mood_tired': 0,
#     'mood_bored': 0,

#     # Context Location (one-hot)
#     'context_location_home': 1,
#     'context_location_commute': 0,
#     'context_location_work': 0,

#     # Companion (one-hot)
#     'companion_alone': 1,
#     'companion_with_family': 0,
#     'companion_with_friends': 0,

#     # Device (one-hot)
#     'device_phone': 1,
#     'device_laptop': 0,
#     'device_tablet': 0,

#     # Demographics (label encoded)
#     'age_group_encoded': 1,      # e.g., 18-25
#     'location_encoded': 3,       # e.g., Urban
#     'diet_encoded': 0,           # e.g., Vegetarian
#     'transport_encoded': 2,      # e.g., Public Transport

#     # Action will be updated dynamically in the loop
#     'recommended_action_encoded': None
# }


# Get the recommended action
recommended_action = get_best_action(user_context, actions, rf)

print(f"Recommended Action: {recommended_action[0]} with predicted acceptance: {recommended_action[1]}")

import joblib

# Save the label encoder for recommended_action
joblib.dump(le5, 'label_encoder.pkl')

print("✅ LabelEncoder saved as 'label_encoder.pkl'")

# Define your actions
actions = ['turn_off_appliances', 'fix_leaky_taps', 'take_shorter_showers',
       'carry_reusable_bag', 'set_ac_to_24_degrees',
       'use_public_transport', 'avoid_single_use_plastic',
       'compost_kitchen_waste', 'buy_local_produce',
       'use_refillable_bottles']  # List all your actions

# Function to get the predicted acceptance for each action
def get_best_action(user_context, actions, model):
    predicted_acceptances = []

    # Iterate through each action and predict acceptance
    for action in actions:    
        encoded_action = le5.transform([action])[0]  # Convert action string to encoded int

        user_context['recommended_action_encoded'] = encoded_action
        
        # Convert user context into DataFrame (same format as your training data)
        users_df = pd.DataFrame([user_context])  # Your user context should be formatted as a DataFrame

        users_df = users_df[encoded_df.columns]

        
        # Predict the probability of acceptance for this action (1 indicates acceptance)
        acceptance_score = model.predict_proba(users_df)[:, 1]  # We use the probability of 'accepted' (class 1)
        
        # Store the action and its predicted acceptance score
        predicted_acceptances.append((action, acceptance_score[0]))

    # Sort the actions by predicted acceptance score (highest first)
    recommended_action = sorted(predicted_acceptances, key=lambda x: x[1], reverse=True)[0]
    
    return recommended_action

# Example user context (this would be provided in real use cases)
user_context = {
    # Time (one-hot, only one should be 1)
    'time_morning': 1,
    'time_afternoon': 0,
    'time_evening': 0,
    'time_night': 0,
    'time_weekend': 0,

    # Goal (one-hot)
    'goal_energy_efficient': 0,
    'goal_plastic_free': 1,
    'goal_reduce_emissions': 0,
    'goal_water_conservation': 0,
    'goal_zero_waste': 0,

    # Weather (one-hot)
    'weather_sunny': 1,
    'weather_cold': 0,
    'weather_hot': 0,
    'weather_rainy': 0,

    # Mood (one-hot)
    'mood_happy': 0,
    'mood_motivated': 1,
    'mood_tired': 0,
    'mood_bored': 0,

    # Context Location (one-hot)
    'context_location_home': 1,
    'context_location_commute': 0,
    'context_location_work': 0,

    # Companion (one-hot)
    'companion_alone': 1,
    'companion_with_family': 0,
    'companion_with_friends': 0,

    # Device (one-hot)
    'device_phone': 1,
    'device_laptop': 0,
    'device_tablet': 0,

    # Demographics (label encoded)
    'age_group_encoded': 1,      # e.g., 18-25
    'location_encoded': 3,       # e.g., Urban
    'diet_encoded': 0,           # e.g., Vegetarian
    'transport_encoded': 2,      # e.g., Public Transport

    # Action will be updated dynamically in the loop
    'recommended_action_encoded': None
}


# Get the recommended action
recommended_action = get_best_action(user_context, actions, rf)

print(f"Recommended Action: {recommended_action[0]} with predicted acceptance: {recommended_action[1]}")

import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

joblib.dump(le1, 'age_group_encoder.pkl')
joblib.dump(le2, 'location_encoder.pkl')
joblib.dump(le3, 'diet_encoder.pkl')
joblib.dump(le4, 'transport_encoder.pkl')

# Load trained model and label encoder (assuming you saved them previously)
rf = joblib.load("random_forest_model.pkl")
le5 = joblib.load("label_encoder.pkl")  # For recommended_action
le_age_group = joblib.load("age_group_encoder.pkl")  # For age_group encoding
le_location = joblib.load("location_encoder.pkl")  # For location encoding
le_diet = joblib.load("diet_encoder.pkl")  # For diet encoding
le_transport = joblib.load("transport_encoder.pkl")  # For transport encoding



print("✅ Model saved as 'random_forest_model.pkl'")

# Define the feature columns (same order as used during training)
feature_columns = [
    'time_afternoon', 'time_evening', 'time_morning', 'time_night', 'time_weekend',
    'goal_energy_efficient', 'goal_plastic_free', 'goal_reduce_emissions', 'goal_water_conservation', 'goal_zero_waste',
    'weather_cold', 'weather_hot', 'weather_rainy', 'weather_sunny',
    'mood_bored', 'mood_happy', 'mood_motivated', 'mood_tired',
    'context_location_commute', 'context_location_home', 'context_location_work',
    'companion_alone', 'companion_with_family', 'companion_with_friends',
    'device_laptop', 'device_phone', 'device_tablet',
    'age_group_encoded', 'location_encoded', 'diet_encoded', 'transport_encoded', 'recommended_action_encoded'
]

# List of recommended actions (can be updated based on your domain logic)
actions = ['turn_off_appliances', 'fix_leaky_taps', 'take_shorter_showers',
           'carry_reusable_bag', 'set_ac_to_24_degrees',
           'use_public_transport', 'avoid_single_use_plastic',
           'compost_kitchen_waste', 'buy_local_produce',
           'use_refillable_bottles']

st.title("🌱 Terrawise: Context-Aware Sustainable Action Recommender")
st.markdown("Provide your current context and we'll recommend the best action!")

# Input widgets for user context
age_group = st.selectbox("Select Age Group", ['18-24', '25-34', '35-44', '45-54', '55+'])
location = st.selectbox("Select Location", ['delhi', 'hyderabad', 'pune', 'kolkata', 'chennai', 'mumbai', 'bangalore'])
diet = st.selectbox("Select Diet", ['veg', 'non_veg', 'vegan'])
transport = st.selectbox("Select Transport Mode", ['car', 'bike', 'public_transport', 'walking', 'rideshare'])
time = st.selectbox("Time of Day", ['morning', 'afternoon', 'evening', 'night', 'weekend'])
weather = st.selectbox("Weather", ['sunny', 'rainy', 'hot', 'cold'])
mood = st.selectbox("Mood", ['happy', 'tired', 'motivated', 'bored'])
context_location = st.selectbox("Where are you?", ['home', 'commute', 'work'])
companion = st.selectbox("Who are you with?", ['alone', 'with_friends', 'with_family'])
device = st.selectbox("Using Device", ['phone', 'laptop', 'tablet'])

# Input widget for sustainability goal
goal = st.selectbox("Select Your Sustainability Goal", ['energy_efficient', 'plastic_free', 'reduce_emissions', 'water_conservation', 'zero_waste'])

submitted = st.button("Get Recommendation")

if submitted:
    # Define one-hot groups
    context = {col: 0 for col in feature_columns}

    # One-hot encoded fields
    context[f"time_{time}"] = 1
    context[f"weather_{weather}"] = 1
    context[f"mood_{mood}"] = 1
    context[f"context_location_{context_location}"] = 1
    context[f"companion_{companion}"] = 1
    context[f"device_{device}"] = 1

    # Label encoded fields
    context['age_group_encoded'] = le_age_group.transform([age_group])[0]
    context['location_encoded'] = le_location.transform([location])[0]
    context['diet_encoded'] = le_diet.transform([diet])[0]
    context['transport_encoded'] = le_transport.transform([transport])[0]

    # Goal-based one-hot encoding (for goal selection)
    context[f"goal_{goal}"] = 1  # Only the selected goal will be 1, the rest will remain 0

    # Loop through actions and predict acceptance
    action_scores = []
    for action in actions:
        context_copy = context.copy()
        context_copy['recommended_action_encoded'] = le5.transform([action])[0]
        input_df = pd.DataFrame([context_copy])[feature_columns]
        acceptance_prob = rf.predict_proba(input_df)[0][1]  # Prob of accepting the action
        action_scores.append((action, acceptance_prob))

    # Recommend highest score action
    action_scores.sort(key=lambda x: x[1], reverse=True)
    best_action, score = action_scores[0]

    st.success(f"🎯 Recommended Action: **{best_action.replace('_', ' ').title()}**")
    st.write(f"Predicted Acceptance Probability: `{round(score * 100, 2)}%`")

    st.markdown("### 📊 Full Predictions")
    st.dataframe(pd.DataFrame(action_scores, columns=["Action", "Acceptance Probability"]).sort_values(by="Acceptance Probability", ascending=False))

