import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score



df = pd.read_csv('data-full.csv')
df = df[df['campaign'] < 33]
df = df[df['previous'] < 31]
bool_columns = ['housing', 'loan', 'y']
for col in  bool_columns:
    df[col+'_new']=df[col].apply(lambda x : 1 if x == 'yes' else 0)
    df.drop(col, axis=1, inplace=True)

df = df.drop(columns=['default'])
month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

# ใช้ method map เพื่อแทนค่าในคอลัมน์ month ด้วยตัวเลขจากพจนานุกรม
df['month'] = df['month'].map(month_mapping)

le_en = LabelEncoder()

le_education = LabelEncoder()
df['job'] = le_education.fit_transform(df['job'])
df["job"].unique()
#le.classes_

le_education = LabelEncoder()
df['marital'] = le_education.fit_transform(df['marital'])
df["marital"].unique()
#le.classes_

le_education = LabelEncoder()
df['education'] = le_education.fit_transform(df['education'])
df["education"].unique()
#le.classes_

le_education = LabelEncoder()
df['contact'] = le_education.fit_transform(df['contact'])
df["contact"].unique()
#le.classes_

le_education = LabelEncoder()
df['poutcome'] = le_education.fit_transform(df['poutcome'])
df["poutcome"].unique()
#le.classes_

X = df.drop("y_new", axis=1)
y = df["y_new"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

model_rf = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=2)
model_rf.fit(X_train,y_train)

# ทดสอบโมเดลบนชุดทดสอบ
y_pred = model_rf.predict(X_test)

# ทดสอบความสำคัญของ feature ด้วยสถิติ
X_with_const = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_with_const).fit()

selected_features = X.columns[model_sm.pvalues[1:] < 0.05]

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

model_selected = RandomForestClassifier()
model_selected.fit(X_train_selected, y_train)

# ทดสอบโมเดลบนชุดทดสอบ
y_pred_selected = model_selected.predict(X_test_selected)


st.title("Model time Deposit by Natthaphat Toichatturat")

st.write("""### We need some information to predict""")
st.write("Attribute information:")

st.write("Input variables:")
   # bank client data:
st.write("1 - age (numeric)")

st.write('3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed) ')
st.write('4 - education (categorical: "unknown","secondary","primary","tertiary")')
st.write('5 - default: has credit in default? (binary: "yes","no")')
st.write('6 - balance: average yearly balance, in euros (numeric) ')
st.write('7 - housing: has housing loan? (binary: "yes","no")')
st.write('8 - loan: has personal loan? (binary: "yes","no")')
   # related with the last contact of the current campaign:
st.write('9 - contact: contact communication type (categorical: "unknown","telephone","cellular") ')
st.write('10 - day: last contact day of the month (numeric)')
st.write('11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")')
st.write('12 - duration: last contact duration, in seconds (numeric)')
   # other attributes:
st.write('13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)')
st.write('14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)')
st.write('15 - previous: number of contacts performed before this campaign and for this client (numeric)')
st.write('16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")')

marital = (
    'married', 'single', 'divorced'
)

education = (
    'tertiary', 'secondary', 'unknown', 'primary'
)

contact = (
    'unknown', 'cellular', 'telephone'
)

poutcome = (
    'unknown', 'failure', 'other', 'success'
)

age = st.slider("Age", 0, 90, 1)
marital = st.selectbox("marital", marital)
education = st.selectbox("Education Level", education)
balance = st.number_input("Balance", step=1, format="%d", value=0, key="age_input")
contact = st.selectbox("contact", contact )
duration = st.number_input("Duration", step=1, format="%d", value=0, help="perSecond" )
campaign = st.number_input("Campaign", step=1, format="%d", value=0,  )
pdays = st.number_input("Pdays", step=1, format="%d", value=0,  )
previous = st.number_input("Previous", step=1, format="%d", value=0,  )
poutcome = st.selectbox("Post outcome", poutcome)
st.write("Have = 1")
housing_new = st.slider("Housing", 0, 1, 0)
loan_new = st.slider("Loan", 0, 1, 0)


ok = st.button("Predict")
if ok:
    X_input = np.array([[age, marital, education,balance,contact, duration,campaign,pdays,previous,poutcome,housing_new,loan_new]])
    X_input[:, 1] = le_en.fit_transform(X_input[:,1])
    X_input[:, 2] = le_en.fit_transform(X_input[:,2])
    X_input[:, 4] = le_en.fit_transform(X_input[:,4])
    X_input[:, 9] = le_en.fit_transform(X_input[:,9])
    X_input = X_input.astype(float)

    y_pred_input = model_selected.predict(X_input)
    # deposit = np.where(y_pred_input > 0.5, 1, 0)
    
    st.subheader(f"This customer trend to have time deposit is {y_pred_input[0]} ")
    st.subheader("0 = No, 1 = Yes")
    
st.title("Analytics this Model")




model_score =cross_val_score(estimator=RandomForestClassifier(),X=X_train, y=y_train, cv=5)
st.write(f"model_score={model_score}")
st.write(f"model_score={model_score.mean()}")

model_param = {
    'RandomForestClassifier':{
        'model':RandomForestClassifier(),
        'param':{
            'n_estimators': [10, 50, 100, 130], 
            'criterion': ['gini', 'entropy'],
            'max_depth': range(2, 4, 1), 
            'max_features': ['auto', 'log2']
        }
    }}

#gridsearch
scores =[]
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(estimator=mp['model'],param_grid=mp['param'],cv=5,return_train_score=False)
    model_selection.fit(X,y)
    scores.append({
        'model': model_name,
        'best_score': model_selection.best_score_,
        'best_params': model_selection.best_params_
    })
    
st.write("### use Gridsearch to find the best model")
    
st.write(scores)



model_rf = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=2)
model_rf.fit(X_train,y_train)

sx = model_rf.score(X_test,y_test)
st.write("model score")
st.write(sx)

#get feature importances from the model
headers = ["name", "score"]
values = sorted(zip(X_train.columns, model_rf.feature_importances_), key=lambda x: x[1] * -1)
rf_feature_importances = pd.DataFrame(values, columns = headers)

#plot feature importances
fig = plt.figure(figsize=(15,7))
x_pos = np.arange(0, len(rf_feature_importances))
plt.bar(x_pos, rf_feature_importances['score'])
plt.xticks(x_pos, rf_feature_importances['name'])
plt.xticks(rotation=90)
plt.title('Feature importances (rf)')
st.title('Feature importances (rf)')
st.pyplot(plt)

# ทดสอบโมเดลบนชุดทดสอบ
y_pred = model_rf.predict(X_test)

# คำนวณ R-squared
r2 = r2_score(y_test, y_pred)
st.write("### R-squared")
st.write(r2)



st.write("### Summary Model")
st.write(model_sm.summary())

st.write("## Select some feature for this model")
st.write("#### for best R-squared values")


# คำนวณ R-squared สำหรับโมเดลที่ถูกเลือก
r2_selected = r2_score(y_test, y_pred_selected)

st.write(f'R-squared before feature selection: {r2}')
st.write(f'R-squared after feature selection: {r2_selected}')

from sklearn.metrics import r2_score

st.write("### Metrics for this model")

y_pred = model_selected.predict(X_test_selected)
r2 = r2_score(y_test, y_pred)
st.write(f'R-squared: {r2}')

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error: {mse}')

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
st.write(f'Mean Absolute Error: {mae}')

# สร้าง DataFrame
df_comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# แสดง DataFrame

st.write(df_comparison)

from sklearn.metrics import confusion_matrix


# สร้าง Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# แสดง Confusion Matrix ในรูปแบบ Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# plt.show()

st.pyplot(plt)

y_pred_binary =y_pred

from sklearn.metrics import f1_score
# คำนวณ F1 Score
f1 = f1_score(y_test, y_pred)
st.write(f'F1 Score: {f1}')

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred_binary)
st.write(f'Accuracy: {accuracy}')

from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred_binary)
st.write(f'Recall: {recall}')

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_pred)
st.write(f'ROC-AUC Score: {roc_auc}')

corr = X_train_selected.corr()
st.write(corr)

st.write("### Summarize")
st.write("##### Analysis:")

st.write("##### The F1 Score is low, indicating problems in both accuracy and comprehensiveness of the model. You may want to consider tuning the thresholds or features used in the model.")
st.write("##### Accuracy High, indicates the accuracy of the model in making predictions.")
st.write("##### High Precision, indicating positive class prediction accuracy.")
st.write("##### Recall is low, indicating the model's coverage ability is still low.")
st.write("##### ROC-AUC Score is high, indicating the discriminative ability of the model.")


df2 = pd.read_csv('data-full.csv')

st.title("Data Analysis")
st.write("## You can read more details in this link")

analyst = "https://www.canva.com/design/DAFx1FYjwiY/3L0ApznwhXXDMYbBV61iPg/edit?utm_content=DAFx1FYjwiY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton"

st.markdown(f"[Full data analyst]({analyst})")

yes_data = df2[df2['y'] == 'yes']
no_data = df2[df2['y'] == 'no']


# สร้างกราฟ
plt.figure(figsize=(10, 6))

# กรองข้อมูล duration ที่ไม่เกิน 3000
yes_data_filtered = yes_data[yes_data['duration'] <= 3000]
no_data_filtered = no_data[no_data['duration'] <= 3000]

max_yes_duration = yes_data['duration'].max()
max_no_duration = no_data['duration'].max()

# ใช้ KDE plot สร้างกราฟความหนาแน่น
sns.kdeplot(yes_data_filtered['duration'], color='blue', label='Yes', fill=True)
sns.kdeplot(no_data_filtered['duration'], color='red', label='No', fill=True)

# หาค่าเฉลี่ยของ Duration สำหรับ y = yes และ no
mean_duration_yes = yes_data['duration'].mean()
mean_duration_no = no_data['duration'].mean()
mean_duration = df2['duration'].mean()

plt.axvline(mean_duration_yes, color='blue', linestyle='dashed', label=f'Mean Duration (Yes) = {mean_duration_yes:.2f}')
plt.axvline(mean_duration_no, color='red', linestyle='dashed', label=f'Mean Duration (No) = {mean_duration_no:.2f}')
plt.axvline(mean_duration, color='green', linestyle='dashed', label=f'Mean Duration = {mean_duration:.2f}')

plt.xlabel('Duration')
plt.ylabel('Density')
plt.title('Density Plot of Duration by Y (Yes/No) (Duration <= 3000)')
plt.legend(loc='upper right')

st.pyplot(plt)
st.write("max Yes =" ,max_yes_duration,"max No =", max_no_duration)
st.write("#### From the graph, it can be seen that after duration > 258, customers will be more interested in opening a time deposit account than not interested. And the red mountain peak is the duration that customers are most interested in time deposits, with a value of approximately 100-200 seconds, which is similar to the duration that customers are most interested in time deposits, or at the blue mountain peak, approximately 200-300. Second, this is called the critical point. It is a point where customers are sensitive in making decisions.")
st.write("#### Therefore, it can be concluded that it is best to contact customers for more than 260 seconds because after more than 260 seconds, customers are more likely to be interested in time deposits and try to make the median duration equal to 426 seconds and the average approximately 537 seconds and at the customer's sensitive point Or the point where the peaks of the red and blue mountains are almost equal, in the range of 100-300 seconds, must make customers pay the most attention to us.")