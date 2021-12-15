# Fundamentals of Big Data Analytics
#####  <div style="text-align: right"> 22181418 물리학과 서진주 </div>
------------
## $\rm D^+$ selection with Boost Decision Tree (BDT)
------------

## Intorduction <br><br>

138억 년 전 우주는 거대한 폭발(Big-Bang)로 시작되었다. 
현대 물리학의 중요한 숙제중의 하나는 이 Big-Bang 직후의 우주 상태를 알아내는 것이다. Big-Bang이 일어난 직후 우주의 에너지 밀도와 온도는 상상을 뛰어넘게 높았을 것이라 예측된다.
우주의 나이가 약 100 만분의 1초 정도였을 때의 우주의 상태를 쿼크-글루온 플라즈마(quark-gluon plasma, 또는 QGP)라고 한다.

우주의 초기 상태로 돌아가 쿼크-글루온 플라즈마를 관측하는 것은 불가능하지만 작은 수의 소립자로 이루어진 계라도 중이온 가속기를 이용하여 초고온의 상태를 만듦으로서 그와 유사한 상태를 만들어, 우주 초기 상태를 연구할 수 있다. 
중이온 가속기를 이용해 무거운 원자핵(중이온)들의 충돌시키면 충돌 중심지역에 생성된 계는 핵자 밀도가 거의 0이며, 초기에는 비평형 상태이나 결국 쿼크-글루온 플라즈마로 진화할 것으로 예상 된다. 
이렇게 충돌 중심지역에 형성된 쿼크-글루온 플라즈마는 급격히 팽창하는데 이 과정에서 온도와 밀도가 한계온도 혹은 한계밀도 이하로 감소하게 되면 강입자로 변환되어 검출기에서 검출 된다.
따라서 중이온 충돌 실험으로부터 생성된 입자의 생성량을 측정하여 쿼크-글루온 플라즈마의 생성 여부를 결정하고 그 특성을 발견할 수 있다.
특히나 섭동 양자 색소 역학(pQCD)에 의하면, 에너지와 운동량을 가진 파톤(parton)이 쿼크-글루온 플라즈마를 통과하면서 잃는 에너지는 쿼크의 색 전하(colour charge)와 질량에 따라 다르게 계산되는데, 이는 아직 실험적으로 검증되지 않았다.
이 때문에 무거운 질량을 가지는 바닥 쿼크(b)와 매력 쿼크(c)의 쿼크-글루온 플라즈마와의 상호작용을 구분하는 일은 RHIC 실험이래로 고에너지 핵물리를 연구하는 학자들에게 중심 과제가 되어왔다. 


------------

## Data sample <br><br>

매력 쿼크를 포함한 중간자인 $\rm D^{+}$는 $\rm K^{-}$ 중간자와 두 개의 $\pi^{+}$중간자로 붕괴한다. 따라서 $\rm K^{-}, \pi^{+}, \pi^{+}$을 조합하면 $\rm D^{+}$를 재구성 할 수 있다.

Machine learning에 사용되는 데이터는 충돌 에너지가 7 TeV 일 때, 2억 4천만개의 양성자-양성자 충돌에서 얻은 입자들이다.
양성자-양성자 충돌 당 대략 5개의 입자들이 발생하므로 총 12억개의 입자들의 정보를 포함하고 있다. 입자들은 전자, 뮤온, 양성자, $\rm K^{-}, \pi^{+}$등이 있다. 7 TeV 양성자-양성자 충돌 시스템에서 생성되는 $\rm D^{+}$의 수는 대략 4천 7백개이다. 12억개의 입자들에서 대략 1만 4찬여개의 입자들을 선택해 $\rm D^{+}$을 재구성해야한다.

data에서는 $\rm K^{-}, \pi^{+}$가 $\rm D^{+}$에서 붕괴되었는지 알 수 없기에 Monte Calro simulation을 통해 입자들이 어디서 붕괴했는지에 대한 정보를 가지고 있는 data sample이 필요하다.

+ Data : 12 Bilions particles without mother particle labeling 
+ MC sample : 4000 particles with mother proticle labeling


------------

## Package and Algorithm <br><br>

### Q2. 
#
+ CPU : Main memory에서 전달되는 명령을 실행하는 부분. 프로그래밍 연산을 담당.
+ Main memory : CPU에 전달할 명령을 저장하는 부분, 휘발성 메모리로 프로그램을 저장함.
+ Second Memory : 운영체제 및 데이터를 저장하는 부분, 비휘발성 메모리로 데이터를 보존할 수 있음.

------------



### Q3. 
#### 1. 
#### 2. 
#### 3. 
#### 4. 
#
The reference code is in [HERE](https://github.com/JinjooSeo/PY4E/blob/main/week1/example/hydro_evo-TestRun.py)

+ 1. Error
    + SyntaxError
     ```Python
     from os import path
     home = path.expanduser("Desktop")
     working_path = path.join(home, "study/PY4E/week1/example")
     data = loadtxt(momentum_anisotropy_eta_-0.5_0.5.dat,dtype=float32) #Invalid syntax
     ```

    + ValueError
     ```Python
     from os import path
     home = path.expanduser("Desktop")
     working_path = path.join(home, "study/PY4E/week1") 
     data = loadtxt("momentum_anisotropy_eta_-0.5_0.5.dat",dtype=int) #The data type of file format and read format is not matched
     ```

------------

### Q4. 
#### hint:

     birth = int(input(“생일이 지났습니까? 맞으면 0 아니면 -1 : “))
#
The macro is in [HERE](https://github.com/JinjooSeo/PY4E/blob/main/week1/M1_A4.py)

```Python
from datetime import datetime
from dateutil.relativedelta import relativedelta

birth = input("When is your birthday? (ex) 2021-07-19) \n")
birth = datetime.strptime(birth, "%Y-%m-%d") #convert to date type(?) from string
today = datetime.now() #Get a date of today automatically

 #Vaildity check
if birth > today: 
    print("Invaild date! Please check the birthday.")
    exit()

#initialization
age_month = 0
age_day = 0

if birth.day <= today.day:
    age_day = today.day - birth.day
else:
    age_day = (today.replace(day=1) - relativedelta(days=1)).day + today.day - birth.day
    today -=relativedelta(months=1)

if birth.month <= today.month:
    age_month = today.month - birth.month
    age_year = today.year - birth.year  
else:
    age_month = 12 + today.month - birth.month
    age_year = today.year - birth.year -1

print(str(age_year)+" years, "+str(age_month)+" months, "+str(age_day)+" days")
print("Your age is " + str(age_year))
```
    
