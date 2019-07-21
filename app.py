from flask import Flask, render_template, request, redirect
import requests
import joblib

app = Flask(__name__)

#### OOS Prediction
## map input
# age
age18_24 = [1,0,0,0,0,0,0]
age25_34 = [0,1,0,0,0,0,0]
age35_44 = [0,0,1,0,0,0,0]
age45_54 = [0,0,0,1,0,0,0]
age55_64 = [0,0,0,0,1,0,0]
age65_older = [0,0,0,0,0,1,0]
age_under18 = [0,0,0,0,0,0,1]
# gender
gender_f = [1,0,0]
gender_m = [0,1,0]
gender_nb = [0,0,1]
# hobby
hobby_y = [1,0]
hobby_n = [0,1]
# hours on computer
hour1_4 = [1,0,0,0,0]
hour5_8 = [0,1,0,0,0]
hour9_12 = [0,0,1,0,0]
hour_lessThan1 = [0,0,0,1,0]
hour_over12 = [0,0,0,0,1]
# hour outside
out1_2 = [1,0,0,0,0]
out3_4 = [0,1,0,0,0]
out30_50mnt = [0,0,1,0,0]
out_30mntLess = [0,0,0,1,0]
out_over4 = [0,0,0,0,1]
# years coding
years0_2 = [1,0,0,0]
years10_orMore = [0,1,0,0]
years2_5 = [0,0,1,0]
years5_10 = [0,0,0,1]
# time fully productive
productive_lessOne = [1,0,0,0]
productive_1_3 = [0,1,0,0]
productive_3_6 = [0,0,1,0]
productive_6_more = [0,0,0,1]
# exercise
exe_1_2_week = [1,0,0,0]
exe_3_4_week = [0,1,0,0]
exe_daily = [0,0,1,0]
exe_no = [0,0,0,1]
# employment
fullTime = [1, 0]
partTime = [0, 1]

## user list
target_name = ['Satisfied', 'Disatisfied', 'Neither satisfied nor dissatisfied']
user = []


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            age = request.form['age']
            gender = request.form['gender']
            hobby = request.form['hobby']
            hourComp = request.form['hourComp']
            hourOut = request.form['hourOut']
            yearsCode = request.form['yearsCoding']
            timeProductive = request.form['timeProductive']
            exercise = request.form['exercise']
            employment = request.form['employment']
            username = request.form['aaaaa']
            user = []

            # age
            if age == 'age1':
                user.extend(age18_24)
            elif age == 'age2':
                user.extend(age25_34)
            elif age == 'age3':
                user.extend(age35_44)
            elif age == 'age4':
                user.extend(age45_54)
            elif age == 'age5':
                user.extend(age65_older)
            else:
                user.extend(age_under18)

            # gender
            if gender == 'gender1':
                user.extend(gender_f)
            elif gender == 'gender2':
                user.extend(gender_m)
            else:
                user.extend(gender_nb)

            # hobby
            if hobby == 'hobby':
                user.extend(hobby_y)
            else:
                user.extend(hobby_n)

            # hour comp
            if hourComp == 'hourComp1':
                user.extend(hour1_4)
            elif hourComp == 'hourComp2':
                user.extend(hour5_8)
            elif hourComp == 'hourComp3':
                user.extend(hour9_12)
            elif hourComp == 'hourComp4':
                user.extend(hour_lessThan1)
            else:
                user.extend(hour_over12)
            
            # hour outside
            if hourOut == 'hourOut1':
                user.extend(out1_2)
            elif hourOut == 'hourOut2':
                user.extend(out3_4)
            elif hourOut == 'hourOut3':
                user.extend(out30_50mnt)
            elif hourOut == 'hourOut4':
                user.extend(out_30mntLess)
            else:
                user.extend(out_over4)

            # years coding
            if yearsCode == 'years1':
                user.extend(years0_2)
            elif yearsCode == 'years2':
                user.extend(years10_orMore)
            elif yearsCode == 'years3':
                user.extend(years2_5)
            else:
                user.extend(years5_10)
            
            # time fully productive
            if timeProductive == 'productive1':
                user.extend(productive_lessOne)
            elif timeProductive == 'productive2':
                user.extend(productive_1_3)
            elif timeProductive == 'productive3':
                user.extend(productive_3_6)
            else:
                user.extend(productive_6_more)

            # exercise
            if exercise == 'exercise1':
                user.extend(exe_1_2_week)
            elif exercise == 'exercise2':
                user.extend(exe_3_4_week)
            elif exercise == 'exercise3':
                user.extend(exe_daily)
            else:
                user.extend(exe_no)

            # employment
            if employment == 'employment1':
                user.extend(fullTime)
            else:
                user.extend(partTime)


            ml = model.predict([user])[0]
            ml_proba = model.predict_proba([user])[0]
            status = target_name[ml]
            percent_status = round(ml_proba[ml], 2) * 100
            probability1 = round(ml_proba[0], 2) * 100
            probability2 = round(ml_proba[1], 2) * 100
            probability3 = round(ml_proba[2], 2) * 100


            userData = {
                'name': username,
                'status': status,
                'probability': percent_status,
                'probability1':probability1,
                'probability2':probability2,
                'probability3':probability3
            }

            print(user)

            output = f'User is {status} with {percent_status} %'

            return render_template ('result.html', userData=userData)
        except:
            return redirect ('error.html')
    else:
        return render_template ('home.html')


@app.route('/result')
def result():
    return render_template ('result.html')


@app.errorhandler(404)
def notFound404(error):
    return render_template('error.html')


if __name__ == '__main__':
    model = joblib.load('stackOverflowSurvey_rfc3_comp')
    app.run(debug=True)