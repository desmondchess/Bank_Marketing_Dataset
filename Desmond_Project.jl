### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ 0fac113a-c4e9-4f70-bcd9-7703c0c39029
begin
	using PlutoUI, CSV, DataFrames, CairoMakie, Random, MLJ, ColorSchemes
	using AlgebraOfGraphics
	import ScikitLearn
end

# ╔═╡ a7957699-cbcc-410b-b649-3c220bc65fe5
import MLJBase: partition, StratifiedCV, train_test_pairs

# ╔═╡ 64f7538b-589b-455e-9f51-895e1be07d06
set_aog_theme!()

# ╔═╡ 68c3fe88-d83f-47cc-ab63-a8d4c474f8dd
TableOfContents()

# ╔═╡ 9ff126c6-18dc-428a-b211-11e48a1d6f03
begin
	ScikitLearn.@sk_import ensemble: RandomForestClassifier
	ScikitLearn.@sk_import linear_model: LogisticRegression
	ScikitLearn.@sk_import svm: SVC
	ScikitLearn.@sk_import naive_bayes:GaussianNB
	ScikitLearn.@sk_import ensemble : IsolationForest
	ScikitLearn.@sk_import metrics: confusion_matrix
	ScikitLearn.@sk_import decomposition: PCA
	ScikitLearn.@sk_import metrics: precision_score
	ScikitLearn.@sk_import metrics: recall_score
end

# ╔═╡ 9226d339-d748-4686-bbab-f3741e211f06
colors = ColorSchemes.tab10

# ╔═╡ 65f9f7d9-0426-40c3-946d-9829704ad0ee
cmap = reverse(ColorSchemes.diverging_gwr_55_95_c38_n256)

# ╔═╡ 1cb4e5c0-6e80-11ed-3097-cff17b15ed78
md"
# 💰 Final Project - Banking Marketing Dataset 


A bank marketing dataset relating to a Portuguese banking institution's direct marketing campaign (phone calls) was downloaded from UCI machine learning. We intend to use this data to predict if a client will subscribe to a term deposit. A term deposit is when a client agrees to lock away their money for an agreed length with a guaranteed interest return rate.

!!! note \"Objective\"
	Our objective is to explore different supervised machine learning algorithms, further explore the support vector machinr algorithms and implement an unsupervised algorithm to help make an inference on the best framework to achieve our project goal.

**💰 the labeled data**

Download and read in the banking marketing dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

!!! note \"Indepth Objectives\"
	We are particularly going to explore more how marital status, job and education impact the client’s response, and if age matters in the response of the client. we believe that starting with this will lead to more interesting gray areas to explore further.


	Using the client attributes, we will train an algorithm that predicts the client’s response to the marketing. For supervised learning, we will explore the following algorithms
	* Support Vector Machines
	* Logistic Regression
	* Naive Bayes
	* Random Forest
	Using the default parameters of these algorithms, we choose the best-performing algorithm and tune the parameters to improve the model. Using the best model, we will conduct a permutation feature importance test to give a score of importance to the attribute that really matters in predicting if a client will subscribe or not.


	Furthermore, we intend to explore unsupervised machine learning algorithms, particularly dimension reduction using diffusion maps to see if we can see a distinct separation of the data in a 2D plane. Using the dimension reduction matrix, we train an anomaly detection algorithm using iForest that creates a decision boundary where the client’s response is defined.

!!! note \"Authors\"
	Gbenga Fabusola, Desmond Obaita
	

"

# ╔═╡ ea11e23d-a413-4b3b-89e4-18f0cbf11145
md"## 💰 Read in and process the banking marketing dataset

Download from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/) and read in the data set, which details the various clients attributes 

"

# ╔═╡ 16e1f373-2409-425a-b98b-ce51ab3ed974
data = CSV.read("bank-additional-full.csv", DataFrame, missingstring="unknown")

# ╔═╡ f54e2fcc-98c8-4732-8efe-08d5f03d2a61
md"
💰 Getting a feel of the total number of clients that agreed to the term deposit and those that did not."

# ╔═╡ f2521124-f78a-4949-9f5c-36383c1df90c
combine(groupby(data, "y"), "y" => (col -> length(col)) => "# of clients")

# ╔═╡ 9627dcfe-f406-4351-8596-ec339bd87fcc
md"
💰 We discovered that there are 10,700 unknowns in our data set. We have no further information about the unknowns and won't be able to run forward with it, so we decided to drop them, considering that we are running this particular data set in all our algorithms and also making a comparison. 

So, even if an algorithm can handle the unknowns, we are not certain if others can handle it the same way or rather, how handling the unknowns changes their end results. We concluded it would be better not including the unknowns."

# ╔═╡ ac8e807b-67f1-4150-9de8-7cbaed48a8ff
dropmissing!(data)

# ╔═╡ 3d960162-89c4-43d9-a7f5-1cd289168d61
data_2 = CSV.read("bank-additional-full.csv", DataFrame, missingstring="unknown")

# ╔═╡ f80c25ab-8cad-4149-b506-620e4c1d90cc
Data_diff = DataFrame("With_Unknowns" => nrow(data_2), "Without_Unknowns" => nrow(data))

# ╔═╡ 6cc8ac36-e5ce-486b-93eb-feab9a322a31
md"
💰 Barplot detailing difference in our Data set with unknowns and without unknowns
"

# ╔═╡ 86b40c24-530b-4105-bdfc-be27592ad906
begin
	color_s = Makie.wong_colors()
	local fig = Figure()
	local ax = Axis(fig[1, 1], xticks = (1:2, ["data_with_Unknowns", "data_without_Unknowns"]), ylabel = "Number of Clients")
	barplot!(1:2, [Data_diff[:,1][1],Data_diff[1,2][1]], color = color_s[1:2])
	fig
end

# ╔═╡ 1f131dd5-d086-4aa8-8aad-82b63f3fb6ad
class_dist = combine(groupby(data, "y"), "y" => (col -> length(col)) => "# of clients")

# ╔═╡ e5f9a2eb-ea32-406d-86cc-4a4c7280a1ef
class_dist[2, "# of clients"]

# ╔═╡ 7e80d3b7-d8e2-40bf-87dc-8af7fb5127fc
ids = shuffle(1:nrow(data))

# ╔═╡ 71b4d250-d7c5-4037-af33-7a9e8272e9c9
md"
💰 Barplot of our class distribution
"

# ╔═╡ c6e2dac0-7505-4c85-ba87-985d2a8312f5
begin
	total_clients = sum(class_dist[:, "# of clients"])
	local fig = Figure()
	local ax = Axis(fig[1, 1], xticks = (1:2, ["no", "yes"]), ylabel = "Number of Clients")
	barplot!(1:2, class_dist[:, "# of clients"]/total_clients, color =color_s[5:6] )
	fig
end

# ╔═╡ fae7618b-7e5f-448b-a89f-9db030b6ef0e
md" ## 💰 Exploring The Data

Our hypothesis is that marital status, age, job and education can be used to distinguish between yes and no desired target (If a client agrees to the term deposit or not) with reasonable accuracy. We could be wrong about this, but all would be made clear as we progress. To test this hypothesis, we draw a bar plot such that:
* each possible feature of each attribute (marital, age, job, education, etc) is listed on the x-axis
* there are two bars side-by-side for each feature of each attribute (marital, age, job, education, etc) : one representing yes, the other representing no.
* the height of the bar represents the number of decisions with that target label _and_ that attribute
* the bars are colored differently according to the desired target
* a legend indicates which color corresponds to which desired target

!!! note \" \"
	we do this using `AlgebraOfGraphics.jl`, an analogous example can be found [here](https://aog.makie.org/stable/generated/penguins/#Styling-by-categorical-variables).

"

# ╔═╡ 6c9d9150-a41e-4999-8a40-76feb71a62ae
begin
	set_aog_theme!()
	local axis = (width = 700, height = 400, title= "Marital Status Distribution", ylabel="# of clients")
	marital_frequency = AlgebraOfGraphics.data(data) * frequency() * mapping(:marital)
	local plt = marital_frequency * mapping(color = :y, dodge = :y)
	draw(plt; axis)	
end

# ╔═╡ 5130601d-f7f2-4ff0-af98-ff113db4069b
md"
💰 Taking a quick peek at the Housing attribute..."

# ╔═╡ 50dc9980-d369-4e87-8edc-979e427066b0
begin
	set_aog_theme!()
	local axis = (width = 700, height = 400, title= "Subscription based on housing status", ylabel="# of clients")
	housing_frequency = AlgebraOfGraphics.data(data) * frequency() * mapping(:housing)
	local plt = housing_frequency * mapping(color = :y, dodge = :y)
	draw(plt; axis)	
end

# ╔═╡ 1f6d1be4-e1ae-4189-bbcb-8324abb10fd7
data

# ╔═╡ 6d377d3d-70f2-41d1-9348-4f921052fa09
md"
💰 Another quick peek at the Monthly Attribute..."

# ╔═╡ 7b2d1427-373e-469d-8fab-23acf7acf1c6
md"
💰 And Another... on loan. All just to get a general overview and test the hypothesis we stated earlier
"

# ╔═╡ 22e8329a-c25b-4ed7-9750-bddd9fbcd133
begin
	set_aog_theme!()
	local axis = (width = 700, height = 400, title= "Subscription based on loan status", ylabel="# of clients")
	loan_frequency = AlgebraOfGraphics.data(data) * frequency() * mapping(:loan)
	local plt = loan_frequency * mapping(color = :y, dodge = :y)
	draw(plt; axis)	
end

# ╔═╡ c62ece86-7de4-4184-a7db-958bf7330975
unique(data[:, "education"])

# ╔═╡ 11502a5b-6352-4533-b9ce-a5b3869ea3e3
begin
	set_aog_theme!()
	local axis = (width = 700, height = 400, title= "Subscription based on client's education", ylabel="# of clients", xticklabelrotation=45)
	edu_frequency = AlgebraOfGraphics.data(data) * frequency() * mapping(:education)
	local plt = edu_frequency * mapping(color = :y, dodge = :y)
	draw(plt; axis)	
end

# ╔═╡ f3651bb3-ef95-4538-bf82-1b0e9151e1ea
begin
	set_aog_theme!()
	local axis = (width = 700, height = 400, title= "Subscription based on client's job", ylabel="# of clients", xticklabelrotation=45)
	job_frequency = AlgebraOfGraphics.data(data) * frequency() * mapping(:job)
	local plt = job_frequency * mapping(color = :y, dodge = :y)
	draw(plt; axis)	
end

# ╔═╡ 76274102-a3a9-445a-9528-15fc4b3bf0e4
md"## 💰 More Viz Into Our Data Sets & Comparisons Between (Yes & No) Clients "

# ╔═╡ 2dc27489-8d68-4c75-9faf-23bed4ce7c8d
begin
	subscribe = filter(:y => n -> n == "yes", data)
	not_subscribe = filter(:y => n -> n == "no", data)
end

# ╔═╡ 356b7832-8217-4d50-87d4-d1c893936bce
eda_ids = shuffle(1:class_dist[2, "# of clients"])[1:50]

# ╔═╡ ac3b425c-4f63-43b8-ab8e-44198f592e7c
begin
	local fig = Figure()
	local ax = Axis(fig[1, 1], xlabel="Age", ylabel="campaign")
	scatter!(subscribe[eda_ids, "age"], subscribe[eda_ids, "campaign"], color=:red, label="subscribe")
	scatter!(not_subscribe[eda_ids, "age"], not_subscribe[eda_ids, "campaign"], color=:blue, label="not subscribe")
	axislegend()
	fig
end

# ╔═╡ d49cf980-d59f-4874-bcea-a2e892b81e92
data

# ╔═╡ ad7385b1-f743-4688-a07f-a5400fcb4bc8
begin
	local fig = Figure()
	local ax = Axis(fig[1, 1], xlabel="Age", ylabel="duration")
	scatter!(subscribe[eda_ids, "age"], subscribe[eda_ids, "duration"], color=:red, label="subscribe")
	scatter!(not_subscribe[eda_ids, "age"], not_subscribe[eda_ids, "duration"], color=:blue, label="not subscribe")
	axislegend()
	fig
end

# ╔═╡ 9d61b9da-3372-43f5-bdbd-1eda12b37ab7
begin
	month_vec = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
	month_order = Dict(month_vec .=> eachindex(month_vec))
end

# ╔═╡ 642ab347-dcc6-499e-b7b2-991c928624e9
data[sortperm(data[:, "month"], by= x->month_order[x]), :]

# ╔═╡ 64deadda-5c4a-4580-a21d-0fc2dd54dddb
begin
	set_aog_theme!()
	local axis = (width = 700, height = 400, title= "Monthly subscription", ylabel="# of clients")
	analysis = data[sortperm(data[:, "month"], by= x->month_order[x]), :]
	month_frequency = AlgebraOfGraphics.data(analysis) * frequency() * mapping(:month)
	local plt = month_frequency * mapping(color = :y, dodge = :y)
	draw(plt; axis)	
end

# ╔═╡ 17e541f3-bbfd-49d9-b234-1c54090c4fd6
begin
	sub_clients = combine(groupby(subscribe, :month), "y" => (col -> length(col)) => "# of clients")
	sub_clients = sub_clients[sortperm(sub_clients[:, "month"], by= x->month_order[x]), :]
	no_sub_clients = combine(groupby(not_subscribe, :month), "y" => (col -> length(col)) => "# of clients")
	no_sub_clients = no_sub_clients[sortperm(no_sub_clients[:, "month"], by= x->month_order[x]), :]
end

# ╔═╡ 9a107ea6-b5dc-4463-9890-21e6387c9f51
begin
	local fig = Figure()
	local ax = Axis(fig[1,1], xlabel = "Month Index", ylabel = "# of clients")
	lines!(1:10, sub_clients[:, "# of clients"], label = "subscription")
	lines!(1:10, no_sub_clients[:, "# of clients"], label = "no subscription")
	axislegend()
	fig
end

# ╔═╡ 1798ba6e-25b2-4244-a486-ab0250b46a05
md"## 💰 Preparing The Features For Supervised Machine Learning

As stated earlier, we would be exploring four supervised machine learning algorithms.

* Support Vector Machines
* Logistic Regression
* Linear Regression
* Random Forest

❗ the `RandomForestClassifier` , 'Linear Regression' , 'Logistic Regression' and 'Support Vector Machines' in scikit-learn does not take a categorical variable as an input if the variable has more than two categories. for example, there are seven unique categories of education. the algorithms implementation cannot handle this. however, the algorithms _can_ handle binary features. so, we will convert each multi-category feature into a set of binary feature. 

_old feature_: education (values it can take on: basic.4y, high.school, basic.6y, professional.course, basic.9y, university.degree, illiterate)

_new features encoding the same information_:
* education_basic.4y (values it can take on: 0, 1)
* education_high.school (values it can take on: 0, 1)
...
* education_illiterate (values it can take on: 0, 1)

"

# ╔═╡ c0769a3c-d931-41ef-aa01-6046fe790bd8
md"
**💰 Now, Let's build Models**
"

# ╔═╡ 33cb6386-572e-4901-ba29-b5fa1afa8452
columns = names(data)

# ╔═╡ ed20ad21-b4f3-4737-9447-261dc435851f
data

# ╔═╡ ae6ef878-5266-4842-bd48-4c7ac8c461a4
num_col = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate",	"cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"]

# ╔═╡ b86b3f96-fed3-48d0-bb1e-f81519a8ef91
md"
💰 Differentiating Between Categorical and Numeric Attributes"

# ╔═╡ 67f38336-3740-48d4-bfde-c7b79393dc32
begin
	cat_data = select(data, Not(num_col))
	
	cat_col = names(select(data, Not(num_col)))
end

# ╔═╡ d7903ff6-9eea-4c5a-88c8-9d57b34309aa
md"""
**💰 Standandizing Numerical Column**
"""

# ╔═╡ 76e46d6a-b022-406e-a026-2eba5462f458
function standardize(data, test_data::DataFrame)
	for col in num_col
		data[:, col] = (data[:, col] .- mean(data[:, col]))./std(data[:, col])
		if test_data
			test_data[:, col] = (test_data[:, col] .- mean(data[:, col]))/std(data[:, col])
		end
	end
	return data, test_data
end

# ╔═╡ 00557c37-a699-4078-a826-b93b17c805f2
md"## 💰 Featurizing The Data To Obtain `X`

An algorithm takes as input a fixed-size vector representation of each example. 


💰 we convert each categorical attribute into a vector and join the results to the numeric attributes. With this, we have an input that can be passed into our algorithms, after, of course you guessed it, converting the results to a matrix first. which we then store (the feature vectors) of the clients in the rows of a (# clients × 57) matrix, `X`. 
"

# ╔═╡ 59b5ed8c-0782-4048-9be5-020f7ffa610e
check_cat = data[:, cat_col]

# ╔═╡ 949d9ac0-5ef6-4c82-9c80-0be9ff20c6a0
for col in names(cat_data)
	if col!= "y"
		unique_features = unique(cat_data[:, col])
		for feature in unique_features
			cat_data[:, "$col $feature"] = Int64.(feature .== cat_data[:, col])
		end
	end
end

# ╔═╡ a91b9dd7-2fbd-4e90-8683-f119cb61469b
num_data = select(cat_data, Not(cat_col))

# ╔═╡ ba3f5188-add3-48a6-baec-d5f11819dc54
begin
	for col in num_col
		num_data[:, col] = data[:, col]
	end
end

# ╔═╡ 4720953c-5e62-49e1-91b8-c57c05ad59f7
md"## 💰 The Target Vector, `y`

💰 Next, we construct the target vector `y` whose element gives `true` if a client agrees to the term deposit and `false` if they didn't. And Yea, a client of `y` must correspond with the row of the feature matrix `X`.

!!! note \" \"
	we basically just pulled a column out of the data frame!
"

# ╔═╡ edc32101-c2ab-47f4-8029-097a7d1fe70e
y = Int64.(num_data[:, "y"].=="yes") 

# ╔═╡ 0436b33f-93e8-410f-9d79-2e4e17db1d61
md"## 💰 Test/Train Split

We will follow the test/train split

we are doing this because our banking data set is imbalanced in terms of the labels (i.e. not a 50/50 split of yes vs. no), we will conduct a _stratified_ split, which intends to preserves the distribution of class labels in the splits of the data.

💰 we will use the `partition` function to create a 80%/20% train/test split of the data. we would therefore account for the splits by storing in `ids_train` and `ids_test` the indices of the data that belong to each split.

"

# ╔═╡ de093b09-36a9-41c8-9ab6-71eeabafa562
begin
	row = size(data)[1]
	ids_train, ids_test = partition(1:row, 0.8, shuffle=true, stratify=y)
end

# ╔═╡ 8f38d211-5104-4437-8ea0-825311c35dec
X = Matrix(select(num_data, Not("y")))

# ╔═╡ f5607870-cab2-40b7-bc16-2db2a2bb973f
for i in 1:size(X)[2]
	x̄ = mean(X[:, i])
	σ = std(X[:, i])
	for j in 1:size(X)[1]
		X[j, i] = (X[j, i] -x̄)/σ
	end
end

# ╔═╡ 4885fe32-18d4-4042-b6a9-10f5b5240be6
X

# ╔═╡ 58158883-4d2e-4187-b4ff-20305bf3486d
begin
	X_train, X_test = X[ids_train, :], X[ids_test, :]
	y_train, y_test = y[ids_train, :], y[ids_test, :]
end

# ╔═╡ 0f5b060d-91bc-433e-8510-d841948637f3
md"## 💰 Overview Of different classification models!!! (without tuning 👀)


"

# ╔═╡ 6f248903-c3d7-4c26-9ab9-ee9bbc47ad13
models = Dict("Logistic Regression" => LogisticRegression(), "Naive Bayes" => GaussianNB(), "RandomForest" => RandomForestClassifier(), "SVM" => SVC())

# ╔═╡ eaf22f25-c3f9-471d-96a8-4b34b358718d
begin
		scores = Dict()
		for (model, key) in zip(values(models), keys(models))
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test)
			scores[key] = (model.score(X_test, y_test), precision_score(y_test, y_pred), recall_score(y_test, y_pred))
		end
end

# ╔═╡ ad6e8c27-47a4-4f4e-9528-98440ead6eb1
scores

# ╔═╡ abc45b2b-8907-4924-b90f-f4569467dfb0
y_base = zeros(Int64, nrow(data))

# ╔═╡ c2ccb540-4a55-4229-96e4-1e8e92675675
sum(y.==y_base)/length(y)

# ╔═╡ 8f5e7e0d-2a14-4e07-85fb-7e78a7a77364
begin
	accs = zeros(length(scores))
	recalls = zeros(length(scores))
	precisions = zeros(length(scores))
	for (i, score) in enumerate(values(scores))
		accs[i] = score[1]
		precisions[i] = score[2]
		recalls[i] = score[3]
	end
end

# ╔═╡ 2ae0a767-7bb6-47c6-8b34-d79c2092e8e3
begin
	local fig = Figure()
	local ax = Axis(fig[1,1],ylabel = "Scores", xticks = (1:3, ["Accuracy", "Precision", "Recall"]),
        title = "Comparison Amongst Our supervised Algorithms")
	
	al = (re = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3,3,3],
	heig = cat(accs, precisions, recalls, dims=(1,1)),
	gr = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
	
	barplot!(al.re, al.heig,
		        dodge = al.gr,
		        color = colors[al.gr])
		       
	label = ["RandomForest", "SVM", "Logistic Regression", "Naive Bayes"]
	element = [PolyElement(polycolor = colors[i]) for i in 1:length(label)]
	
	title_ = "Supervised Algorithms"

	Legend(fig[1,2], element, label, title_)
	fig
end

# ╔═╡ 370b79e7-2fb3-4a8e-8b6f-fbeb72c6912a
md"""
**💰 We decided to further experiment with SVM algorithm**
"""

# ╔═╡ 681aff38-2882-4b9a-8896-1ee22e4c6b22
md"""
## 💰 Support Vector Machine Algorithm - Further Experimenting -  (With Tuning 🔧 & K Cross Validation)

"""

# ╔═╡ e5a52f5b-4af8-46f3-8cf5-a8358a0ae4b0
stratified_cv = StratifiedCV(; nfolds=5,
                               shuffle=true,
                               rng=Random.GLOBAL_RNG)

# ╔═╡ ec1aa73a-ef19-44b5-823b-88a891d790d2
cv_ids = train_test_pairs(stratified_cv, 1:size(X_train)[1], y_train)

# ╔═╡ 8f4d59ad-1b13-4488-ac80-27aff6d39b9b
md" ## 💰 Optimal C
**💰 Now, we try to figure out an optimal C for our support vector machine by looping through a range of values and determining the corresponding precision and recall for each C for further analysis
"

# ╔═╡ b7be4ae4-3fe3-43d5-8402-da2bfa27afb8
cs = [10.0^-i for i in -6:0]

# ╔═╡ 69e3391c-dbd3-430d-928c-cdaaa4e937a2
begin
	precision_kf = zeros(length(cs))
	recall_kf = zeros(length(cs))
	nfolds = length(cv_ids)
	for (i, c) in enumerate(cs)
		recall = zeros(nfolds)
		precision = zeros(nfolds)
		algo = SVC(kernel ="rbf", C=c)
		for (ik, (id_train, id_val)) in enumerate(cv_ids)
			algo.fit(X[id_train, :], y[id_train])
			y_pred1 = algo.predict(X[id_val, :])
			precision[ik] = precision_score(y[id_val], y_pred1)
			recall[ik] = recall_score(y[id_val], y_pred1)
		end
		precision_kf[i] = mean(precision)
		recall_kf[i] = mean(recall)
	end
end

# ╔═╡ c0bbac36-59a5-450e-a6dd-4f8918fa4815
precision_kf

# ╔═╡ b2421d68-ffd0-4ea0-b4db-ac0fbcc765f4
recall_kf

# ╔═╡ acac6263-20fa-406c-86c8-5bdb7e30a2c0
begin
	local fig = Figure()
	local ax = Axis(fig[1,1], xlabel="log(c)", ylabel= "score")
	lines!(ax, log10.(cs), precision_kf, label="precision")
	lines!(ax, log10.(cs), recall_kf, label="recall")
	vlines!([2.6], linestyle=:dash, color=:red)
	axislegend()
	fig
end

# ╔═╡ 2cacaf8c-0c17-442a-8c9e-999d7d9107f0
begin
	clf = SVC(kernel ="rbf", C=10^3)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	
	keep_check = DataFrame("Accuracy" => clf.score(X_test, y_test), "Precision" => precision_score(y_test, y_pred), "Recall" => recall_score(y_test, y_pred), "Algo" => "Supervise")
end

# ╔═╡ 25938291-9a12-41d2-b6c0-0f11da73fea7
md"
💰 With the aid of a confusion matrix, we can compute the Recall, Precision and Specificity of our algorithm for more evaluations.

!!! note \" \"
	More tuning is required here on our SVM algorithmn due of our class imbalance to get better data for our confusion matrix. we do this by specifying `class_weight` and `sample_weight` parameters, further reading can be done [here](https://scikit-learn.org/stable/modules/svm.html). 
"

# ╔═╡ 51b5c125-6e13-4f1e-a53f-73614e4e70e4
## class weight and sample weight

# ╔═╡ fe90daae-fa97-4f6d-969b-35a19b9c4f4f
conf_matrix_SVM = confusion_matrix(y_test,y_pred)

# ╔═╡ 3cb77ab4-10da-4028-b29a-50cf021f8e49
function viz_confusion(cm::Matrix; agree = "Yes")
	cm_to_plot = reverse(cm, dims=1)'

	fig = Figure()
	ax  = Axis(fig[1, 1], 
		xlabel="prediction", ylabel="truth",
		xticks=(1:2, ["No", "$agree"]),
		yticks=(1:2, reverse(["No", "$agree"]))
	)
	hm = heatmap!(cm_to_plot, 
		colormap=ColorSchemes.algae, 
		colorrange=(0, maximum(cm))
	)
	for i = 1:2
        for j = 1:2
            text!("$(Int(cm_to_plot[i, j]))",
                  position=(i, j), align=(:center, :center), color="white", 
				  textsize=50
			)
        end
    end
    Colorbar(fig[1, 2], hm, label="# clients")
	return fig
end

# ╔═╡ 328fd458-1311-42c3-a253-250d6b86c894
viz_confusion(conf_matrix_SVM)

# ╔═╡ 607440f5-c9bf-45ef-80b9-c7f4bc065975
md"
 💰 Recall

which is also known as sensitivity or the true positive rate, tells us how often the model chooses the positive class when the observation is in fact in the positive class. It is calculated by dividing the number of true positives in the matrix by the total number of real positives in the data.

!!! note 
	
	💰 From our analysis, Recall for SVM is 0.4896 (378/772), meaning that the model correctly predicts that a client will agree to the deposit 49% of the time.
	"

# ╔═╡ 9c9e402d-00ad-4708-8010-63cd6cc241ea
md"
💰 PRECISION

Precision measures how often a model is correct when it predicts the positive class. It is calculated by dividing the number of true positives in the matrix by the total number of predicted positives. 

!!! note
	
	💰 From our analysis, Recall for SVM is 0.474 (378/797), when the model predicted a positive class, it was correct 47.4% of the time.
	"

# ╔═╡ 783d7e9d-42f1-44e2-8631-d0ea3e3d68fb
md"
 💰 SPECIFICITY

which is also known as the true negative rate, specificity measures how often the model chooses the negative class when the observation is in fact in the negative class. It is calculated by dividing the number of true negatives by the total number of real negatives in the data. 

!!! note
	
	💰 From our analysis, Specificity for SVM is 0.921 (4907/5326), meaning that the model correctly predicts that a client will not accept the term deposit 92.1% of the time."

# ╔═╡ 774c6377-97ed-4905-a2b0-334b07511935
md"""
## 💰 Permutation Importance

"""

# ╔═╡ 1492011c-d688-434e-96ff-30e071311e0a
	ScikitLearn.@sk_import inspection : permutation_importance


# ╔═╡ b986d9ea-f4e2-4254-82aa-e56a1e07053b
per_imp = permutation_importance(clf, X_test, y_test, n_repeats = 2, random_state = 0 )

# ╔═╡ ee1bec3a-ed01-4c0e-9d60-74a86d3ef1e4
length(per_imp["importances_std"])

# ╔═╡ 3434042a-3dc4-4940-810c-f0efb9cc9ef3
new_per_imp = sum(per_imp["importances"], dims = 2)

# ╔═╡ 31c499b6-dd1b-4772-b106-2f0df2ebfe41
begin
	emp = []
	for i in sortperm(per_imp["importances_mean"])
		if per_imp["importances_mean"][i] - 2 * per_imp["importances_std"][i] > 0
			push!(emp, (names(num_data, Not(["y"]))[i]))
		end
	end
end

# ╔═╡ 26048e90-3ad5-412d-bfa7-21e5085d3f43
begin
	permut_imp = zeros(length(new_per_imp))
	for (index,per_imp) in enumerate(new_per_imp)
		permut_imp[index] = per_imp
	end
end

# ╔═╡ 8957c6eb-952f-42d7-9ca1-f772e2ecf394
names(num_data, Not(["y"]))[45]

# ╔═╡ 1e304e6a-4248-45ee-9fa9-eaccaeaf8127
begin
	import_feat_name = names(num_data, Not("y"))
	import_feat_df = DataFrame("feat_name" => import_feat_name, "perm_importance_score" => permut_imp)
end

# ╔═╡ fce06c4a-d80e-43d6-8802-76038d93267f
top10_most_imp_feat = import_feat_df[partialsortperm(import_feat_df.perm_importance_score, 1:10, rev=true), :]

# ╔═╡ 54d76b66-0475-48df-8655-4c362220e1e9
begin
	
	local fig = Figure()
	xs = top10_most_imp_feat[:,2]
	ys = 1:length(top10_most_imp_feat[:,2])
	local ax = Axis(fig[1,1], title = "Ten Most Important Features", ylabel = "The ten most important features", xlabel = "the importance score of those features (permutation importance)")
	barplot!(ax, ys, xs, color=xs, direction=:x, bar_labels=top10_most_imp_feat[:,1])
	xlims!(0,0.25)
	fig
	
end

# ╔═╡ ad1760f5-422b-4f9a-bbad-2035a8902f15
md"## 💰 Unsupervised Machine Learning Algorithm

PCA Analysis on our data set

"

# ╔═╡ 455e44c2-97b9-42ee-b2d8-8951b1991f7c
function PCA_analysis(X::Matrix)
	for i in 1:size(X)[2]
		x̄ = mean(X[:, i])
		σ = std(X[:, i])
		X[:, i] = (X[:, i] .- x̄)/σ
	end
	pca = PCA(n_components=2)
	# pca.fit(X)
	transformed_pca = pca.fit_transform(X)
	variance = pca.explained_variance_ratio_
	return transformed_pca, variance
end

# ╔═╡ 3f93d7f0-4eec-4287-84f6-b37058c65dc8
transformed_pca, variance = PCA_analysis(X)

# ╔═╡ 836f9e56-fcad-4a0e-8a7e-6b31a6a2582d
sum(variance)

# ╔═╡ 0a7f33bf-72fb-40dd-83be-91fc83e7064f
transformed_pca

# ╔═╡ 5b78aa8a-33ce-4ebe-a451-1fbf3cf96c1d
colors

# ╔═╡ bf9ac14b-3b09-40fa-a70c-e372ada461ba
color_index = Int64.((data[ids, "y"] .== "yes")) .+ 1

# ╔═╡ e5d07d23-aaa8-415b-9cb4-fb898c7db690
begin
	local fig = Figure()
	local ax=Axis(fig[1,1], xlabel="principal component 1", 
				   ylabel="principal component 2")

	scatter!(transformed_pca[ids, 1], transformed_pca[ids, 2],
			 strokecolor=colors[color_index], strokewidth= 1, 
		     transparency=true, 
			 color=(:white, 0))

	yes = [MarkerElement(color = colors[2] , marker = 'O', markersize = 15)]
	no = [MarkerElement(color = colors[1] , marker = 'O', markersize = 15)]
	
	Legend(fig[1, 2], [yes, no], ["Yes", "No"], patchsize = (35, 35), rowgap = 10)
	
	fig
end

# ╔═╡ 46d24ac1-ba76-4574-ad01-052f2675794b
md"## 💰 Anomaly Dection using iForest

* An anomaly will be that the client subscribed to the service

"


# ╔═╡ e763fc92-9015-4cd1-9dbb-61e4488a1573
anoms = IsolationForest(random_state=0).fit(transformed_pca)

# ╔═╡ f30ddb52-f36b-4051-a1ff-cc33b3fd8465
anomaly_scores = anoms.decision_function(transformed_pca)

# ╔═╡ f9d4b37b-1687-484e-8225-621fbd4b93eb
begin
	local fig = Figure()
	local ax = Axis(fig[1, 1], xlabel="anomaly_scores", ylabel="no of sample")
	vlines!(0.0, linestyle="--", color=:black)
	density!(anomaly_scores)
	fig
end

# ╔═╡ d80aebe1-a202-4683-9507-902a01020c64
X_tuple = (
        range(minimum(transformed_pca[:, 1]), maximum(transformed_pca[:, 1]), length=100), 
	    range(minimum(transformed_pca[:, 2]), maximum(transformed_pca[:, 2]), length=100)
		  )

# ╔═╡ 341c57be-e3fc-4490-85f3-ccf8ca9e63ca
begin
	X1, X2 = X_tuple
	feature_space = Matrix(DataFrame("x1"=>X1, "x2"=>X2))

	feature_matrix = zeros(length(X1), length(X2))
	for i in 1:length(X1)
		for j in 1:length(X2)
			matrix_space = [feature_space[i, 1] feature_space[j, 2]]
			feature_matrix[i, j] = anoms.decision_function(matrix_space)[1]
		end
	end
end

# ╔═╡ 92c43f2a-1acb-4f84-af1c-18a03f519eef
feature_matrix

# ╔═╡ 86879971-6b68-4b79-84ed-0681aecc483d
anoms_index = ids[1:30]

# ╔═╡ 8bfc87d1-c577-435c-ad98-a2b07b52bf55
Int.(data[anoms_index, "y"].=="yes") .+ 1

# ╔═╡ 3eaf02ff-6c9f-49c9-a421-b7803e9ccf04
function plot_anomaly(X, feature_matrix=feature_matrix)
	local fig = Figure()
	local ax = Axis(fig[1, 1], xlabel="Principal Component 1", ylabel="Principal Component 2", title="Anomaly Score Grid")
	
	
	X1, X2= X
	sep = Int.(data[anoms_index, "y"].=="yes") .+ 1
	colors_anoms = [:blue, :red]
	
	limit = (-maximum(abs.(feature_matrix)), maximum(abs.(feature_matrix)))
	hm = heatmap!(X1, X2, feature_matrix, colormap=cmap, colorrange=limit)
	Colorbar(fig[1, 2], hm, label="anomaly score")
	contour!(X1, X2, feature_matrix, levels=[0.0], color=:black)

	
	
	scatter!(transformed_pca[anoms_index, 1], transformed_pca[anoms_index, 2], color=colors_anoms[sep])
	
	fig
end

# ╔═╡ da1fe9ff-9ba4-46a4-b6fa-d2764a705767
colors

# ╔═╡ 599f476d-a25a-4b28-8f98-75749d489333
plot_anomaly(X_tuple)

# ╔═╡ 05e03eee-6bf0-40cf-a547-f39e8a2aba1d
begin
	y_pred_anoms = Int64.(anomaly_scores .< 0)
	y_true_anoms = Int64.(data[:, "y"].=="yes")
end

# ╔═╡ 85175006-4417-441a-8b79-13a153ebf6f5
Int64.(anomaly_scores .< 0)

# ╔═╡ 61dbb38e-28bd-4b66-9985-da32902e2330
anoms_matrix = confusion_matrix(y_true_anoms, y_pred_anoms)

# ╔═╡ 61e82596-ead5-45c9-9dd7-985dec241ab7
accuracy = sum(y_true_anoms.==y_pred_anoms)/length(y_true_anoms)

# ╔═╡ 5252bbc4-8fc2-4226-9fa6-9f879158649f
begin
	precision = precision_score(y_true_anoms, y_pred_anoms)
	recall = recall_score(y_true_anoms, y_pred_anoms)
end

# ╔═╡ 26e32d31-0a71-4863-9fe3-fb42ad46837b
keep_check

# ╔═╡ 88e94319-09ae-4f16-98e2-70ff0f7d58a9
check = push!(keep_check, [accuracy, precision, recall, "Unsupervised"])

# ╔═╡ e14f1d9c-5351-4fbb-b8a1-d6f24d59dd1b
viz_confusion(anoms_matrix)

# ╔═╡ 2198ef09-bd59-4357-b259-ce8d3debcfca
begin
	local fig = Figure()
	local ax = Axis(fig[1,1], xticks = (1:3, ["Precision", "Recall", "Accuracy"]),
        title = "Comparison Between supervised and unsupervised algorithm")
	
	ol = (re = [1, 1, 2, 2, 3, 3],
		heig = cat(check[:, "Accuracy"], check[:, "Precision"], check[:, "Recall"], dims = (1, 1)),
		 gr = [1, 2, 1, 2, 1, 2])
	
	barplot!(ol.re, ol.heig,
		        dodge = ol.gr,
		        color = colors[ol.gr])
		       
	labels = ["supervised", "unsupervised"]
	elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
	
	title = "Algorithms"

	Legend(fig[1,2], elements, labels, title)
	fig
end

# ╔═╡ 302c45b9-a000-404c-837c-e3bdc062b8eb
md" ## 💰 Conclusion 

"

# ╔═╡ eefd2503-9019-4cec-916f-d104590c9de4
md"
Based on our experiment, supervised learning proves to be the best model to predict the behaviour of our clients
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJBase = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ScikitLearn = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"

[compat]
AlgebraOfGraphics = "~0.6.12"
CSV = "~0.10.7"
CairoMakie = "~0.9.3"
ColorSchemes = "~3.20.0"
DataFrames = "~1.4.3"
MLJ = "~0.19.0"
MLJBase = "~0.21.2"
PlutoUI = "~0.7.48"
ScikitLearn = "~0.6.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.1"
manifest_format = "2.0"
project_hash = "f8a58487ab3b43c870e67e6e24b7c2754e13b9e7"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "e8c8e0a2be6eb4f56b1672e46004463033daa409"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.4.1"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "52b3b436f8f73133d7bc3a6c71ee7ed6ab2ab754"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.3"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.AlgebraOfGraphics]]
deps = ["Colors", "Dates", "Dictionaries", "FileIO", "GLM", "GeoInterface", "GeometryBasics", "GridLayoutBase", "KernelDensity", "Loess", "Makie", "PlotUtils", "PooledArrays", "RelocatableFolders", "StatsBase", "StructArrays", "Tables"]
git-tree-sha1 = "f4d6d0f2fbc6b2c4a8eb9c4d47d14b9bf9c43d23"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.6.12"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "c5fd7cd27ac4aed0acf4b73948f0110ff2a854b2"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.7"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA", "SnoopPrecompile"]
git-tree-sha1 = "5e2c8d04a4b98f73da6d314f673c8b7e7db2d76d"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.9.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "5084cc1a28976dd1642c9f337b28a3cb03e0f7d2"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.7"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes", "UnicodePlots"]
git-tree-sha1 = "23fe4c6668776fedfd3747c545cd0d1a5190eb15"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.9"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "e08915633fcb3ea83bf9d6126292e5bc5c739922"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.13.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d4f69885afa5e6149d0cab3818491565cf41446d"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.4.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "e82c3c97b5b4ec111f3c1b55228cebc7510525a2"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.25"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "a7756d098cbabec6b3ac44f369f74915e8cfd70a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.79"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c36550cb29cbe373e95b3f40486b9a4148f89ffd"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "802bfc139833d2ba893dd9e62ba1767c88d708ae"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.5"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "38a92e40157100e796690421e34a11c107205c86"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "884477b9886a52a84378275737e2823a5c98e349"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.1"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "fe9aea4ed3ec6afdfbeb5a4f39a2208909b162a6"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.5"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "678d136003ed5bceaab05cf64519e3f956ffa4ba"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.9.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "e1acc37ed078d99a714ed8376446f92a5535ca65"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.5.5"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "0cf92ec945125946352f3d46c96976ab972bde6f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.3.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "842dd89a6cb75e02e85fdd75c760cdc43f5d6863"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.6"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "d7df9a6fdd82a8cfdfe93a94fcce35515be634da"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.3"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "42938ab65e9ed3c3029a8d2c58382ca75bdab243"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.8.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "46efcea75c890e5d820e670516dc156689851722"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.4"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "946607f84feb96220f480e0422d3484c49c00239"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.19"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LossFunctions]]
deps = ["InteractiveUtils", "Markdown", "RecipesBase"]
git-tree-sha1 = "53cd63a12f06a43eef6f4aafb910ac755c122be7"
uuid = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
version = "0.8.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "LinearAlgebra", "MLJBase", "MLJEnsembles", "MLJIteration", "MLJModels", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "ScientificTypes", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "9d79ef8684eb15a6fe4c3654cdb9c5de4868a81e"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.19.0"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LinearAlgebra", "LossFunctions", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "ScientificTypes", "Serialization", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "decaf881165c0b3c7abf1130dfe3221ee88ef99a"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "0.21.2"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJBase", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatsBase"]
git-tree-sha1 = "bb8a1056b1d8b40f2f27167fc3ef6412a6719fbf"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.3.2"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "be6d5c71ab499a59e82d65e00a89ceba8732fcd5"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.5.1"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "c8b7e632d6754a5e36c0d94a4b466a5ba3a30128"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.8.0"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Combinatorics", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "RelocatableFolders", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "08203fc87a7f992cee24e7a1b2353e594c73c41c"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.16.2"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase"]
git-tree-sha1 = "02688098bd77827b64ed8ad747c14f715f98cfc4"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.7.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "MiniQhull", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "bdbc2178494c5328defb095a04d553278705b448"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.18.4"

[[deps.MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "c1885d865632e7f37e5a1489a164f44c54fb80c9"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.5.2"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.MarchingCubes]]
deps = ["SnoopPrecompile", "StaticArrays"]
git-tree-sha1 = "ffc66942498a5f0d02b9e7b1b1af0f5873142cdc"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test", "UnicodeFun"]
git-tree-sha1 = "f04120d9adf4f49be242db0b905bea0be32198d1"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.4"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.MiniQhull]]
deps = ["QhullMiniWrapper_jll"]
git-tree-sha1 = "9dc837d180ee49eeb7c8b77bb1c860452634b0d1"
uuid = "978d7f02-9e05-4691-894f-ae31a51d76ca"
version = "0.4.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "5ae7ca23e13855b3aba94550f26146c01d259267"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "f71d8950b724e9ff6110fc948dff5a329f901d64"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg"]
git-tree-sha1 = "88dfa70c818f7a4728c6b82a72a0e597e083938b"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.3.0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "df6830e37943c7aaa10023471ca47fb3065cc3c4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6e9dba33f9f2c44e08a020b0caf6903be540004"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.19+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "84a314e3926ba9ec66ac097e3635e270986b0f10"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.9+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "b64719e8b4504983c7fca6cc9db3ebc8acc2a4d6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "5b7690dd212e026bbab1860016a6601cb077ab66"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "4be53d093e9e37772cc89e1009e8f6ad10c4681b"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "96f6db03ab535bdb901300f88335257b0018689d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "53b8b07b721b77144a0fbbbc2675222ebf40a02d"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.94.1"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QhullMiniWrapper_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Qhull_jll"]
git-tree-sha1 = "607cf73c03f8a9f83b36db0b86a3a9c14179621f"
uuid = "460c41e3-6112-5d7f-b78c-b6823adb3f2d"
version = "1.0.0+1"

[[deps.Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "238dd7e2cc577281976b9681702174850f8d4cbc"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "8.0.1001+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "18c35ed630d7229c5584b945641a73ca83fb5213"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
git-tree-sha1 = "bc12e315740f3a36a6db85fa2c0212a848bd239e"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.2"

[[deps.ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "2436b15f376005e8790e318329560dcc67188e84"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.3"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "75ccd10ca65b939dab03b812994e571bf1e3e1da"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.2"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.ScikitLearn]]
deps = ["Compat", "Conda", "DataFrames", "Distributed", "IterTools", "LinearAlgebra", "MacroTools", "Parameters", "Printf", "PyCall", "Random", "ScikitLearnBase", "SparseArrays", "StatsBase", "VersionParsing"]
git-tree-sha1 = "de6a32950c170e5fd5a2d8bcba0fb97a1028ab06"
uuid = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
version = "0.6.5"

[[deps.ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "efd23b378ea5f2db53a55ae53d3133de4e080aa9"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.16"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StableRNGs]]
deps = ["Random", "Test"]
git-tree-sha1 = "3be7d49667040add7ee151fefaf1f8c04c8c8276"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.0"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "ffc098086f35909741f71ce21d03dadf0d2bfa76"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.11"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "ab6083f09b3e617e34a956b43e9d51b824206932"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.1.1"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "a5e15f27abd2692ccb61a99e0854dfb7d48017db"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.33"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "13237798b407150a6d2e2bce5d793d7d9576e99e"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.13"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "f8cd5b95aae14d3d88da725414bdde342457366f"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.2"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "e4bdc63f5c6d62e80eb1c0043fcc0360d5950ff7"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.10"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["ColorSchemes", "ColorTypes", "Contour", "Crayons", "Dates", "FileIO", "FreeType", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "Requires", "SnoopPrecompile", "SparseArrays", "StaticArrays", "StatsBase", "Unitful"]
git-tree-sha1 = "e20b01d50cd162593cfd9691628c830769f68987"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "3.3.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d670a70dd3cdbe1c1186f2f17c9a68a7ec24838c"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.12.2"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╠═0fac113a-c4e9-4f70-bcd9-7703c0c39029
# ╠═a7957699-cbcc-410b-b649-3c220bc65fe5
# ╠═64f7538b-589b-455e-9f51-895e1be07d06
# ╠═68c3fe88-d83f-47cc-ab63-a8d4c474f8dd
# ╠═9ff126c6-18dc-428a-b211-11e48a1d6f03
# ╠═9226d339-d748-4686-bbab-f3741e211f06
# ╠═65f9f7d9-0426-40c3-946d-9829704ad0ee
# ╟─1cb4e5c0-6e80-11ed-3097-cff17b15ed78
# ╟─ea11e23d-a413-4b3b-89e4-18f0cbf11145
# ╠═16e1f373-2409-425a-b98b-ce51ab3ed974
# ╟─f54e2fcc-98c8-4732-8efe-08d5f03d2a61
# ╠═f2521124-f78a-4949-9f5c-36383c1df90c
# ╟─9627dcfe-f406-4351-8596-ec339bd87fcc
# ╠═ac8e807b-67f1-4150-9de8-7cbaed48a8ff
# ╠═3d960162-89c4-43d9-a7f5-1cd289168d61
# ╟─f80c25ab-8cad-4149-b506-620e4c1d90cc
# ╟─6cc8ac36-e5ce-486b-93eb-feab9a322a31
# ╠═86b40c24-530b-4105-bdfc-be27592ad906
# ╠═1f131dd5-d086-4aa8-8aad-82b63f3fb6ad
# ╠═e5f9a2eb-ea32-406d-86cc-4a4c7280a1ef
# ╠═7e80d3b7-d8e2-40bf-87dc-8af7fb5127fc
# ╟─71b4d250-d7c5-4037-af33-7a9e8272e9c9
# ╠═c6e2dac0-7505-4c85-ba87-985d2a8312f5
# ╟─fae7618b-7e5f-448b-a89f-9db030b6ef0e
# ╠═6c9d9150-a41e-4999-8a40-76feb71a62ae
# ╟─5130601d-f7f2-4ff0-af98-ff113db4069b
# ╠═50dc9980-d369-4e87-8edc-979e427066b0
# ╠═1f6d1be4-e1ae-4189-bbcb-8324abb10fd7
# ╠═642ab347-dcc6-499e-b7b2-991c928624e9
# ╟─6d377d3d-70f2-41d1-9348-4f921052fa09
# ╠═64deadda-5c4a-4580-a21d-0fc2dd54dddb
# ╟─7b2d1427-373e-469d-8fab-23acf7acf1c6
# ╠═22e8329a-c25b-4ed7-9750-bddd9fbcd133
# ╠═c62ece86-7de4-4184-a7db-958bf7330975
# ╠═11502a5b-6352-4533-b9ce-a5b3869ea3e3
# ╠═f3651bb3-ef95-4538-bf82-1b0e9151e1ea
# ╟─76274102-a3a9-445a-9528-15fc4b3bf0e4
# ╠═2dc27489-8d68-4c75-9faf-23bed4ce7c8d
# ╠═356b7832-8217-4d50-87d4-d1c893936bce
# ╠═ac3b425c-4f63-43b8-ab8e-44198f592e7c
# ╠═d49cf980-d59f-4874-bcea-a2e892b81e92
# ╠═ad7385b1-f743-4688-a07f-a5400fcb4bc8
# ╠═9d61b9da-3372-43f5-bdbd-1eda12b37ab7
# ╠═17e541f3-bbfd-49d9-b234-1c54090c4fd6
# ╠═9a107ea6-b5dc-4463-9890-21e6387c9f51
# ╟─1798ba6e-25b2-4244-a486-ab0250b46a05
# ╟─c0769a3c-d931-41ef-aa01-6046fe790bd8
# ╠═33cb6386-572e-4901-ba29-b5fa1afa8452
# ╠═ed20ad21-b4f3-4737-9447-261dc435851f
# ╠═ae6ef878-5266-4842-bd48-4c7ac8c461a4
# ╟─b86b3f96-fed3-48d0-bb1e-f81519a8ef91
# ╠═67f38336-3740-48d4-bfde-c7b79393dc32
# ╟─d7903ff6-9eea-4c5a-88c8-9d57b34309aa
# ╠═76e46d6a-b022-406e-a026-2eba5462f458
# ╟─00557c37-a699-4078-a826-b93b17c805f2
# ╠═59b5ed8c-0782-4048-9be5-020f7ffa610e
# ╠═949d9ac0-5ef6-4c82-9c80-0be9ff20c6a0
# ╠═a91b9dd7-2fbd-4e90-8683-f119cb61469b
# ╠═ba3f5188-add3-48a6-baec-d5f11819dc54
# ╟─4720953c-5e62-49e1-91b8-c57c05ad59f7
# ╠═edc32101-c2ab-47f4-8029-097a7d1fe70e
# ╟─0436b33f-93e8-410f-9d79-2e4e17db1d61
# ╠═de093b09-36a9-41c8-9ab6-71eeabafa562
# ╠═8f38d211-5104-4437-8ea0-825311c35dec
# ╠═f5607870-cab2-40b7-bc16-2db2a2bb973f
# ╠═4885fe32-18d4-4042-b6a9-10f5b5240be6
# ╠═58158883-4d2e-4187-b4ff-20305bf3486d
# ╟─0f5b060d-91bc-433e-8510-d841948637f3
# ╠═6f248903-c3d7-4c26-9ab9-ee9bbc47ad13
# ╠═eaf22f25-c3f9-471d-96a8-4b34b358718d
# ╠═ad6e8c27-47a4-4f4e-9528-98440ead6eb1
# ╠═abc45b2b-8907-4924-b90f-f4569467dfb0
# ╠═c2ccb540-4a55-4229-96e4-1e8e92675675
# ╠═8f5e7e0d-2a14-4e07-85fb-7e78a7a77364
# ╠═2ae0a767-7bb6-47c6-8b34-d79c2092e8e3
# ╟─370b79e7-2fb3-4a8e-8b6f-fbeb72c6912a
# ╟─681aff38-2882-4b9a-8896-1ee22e4c6b22
# ╠═e5a52f5b-4af8-46f3-8cf5-a8358a0ae4b0
# ╠═ec1aa73a-ef19-44b5-823b-88a891d790d2
# ╟─8f4d59ad-1b13-4488-ac80-27aff6d39b9b
# ╠═b7be4ae4-3fe3-43d5-8402-da2bfa27afb8
# ╠═69e3391c-dbd3-430d-928c-cdaaa4e937a2
# ╠═c0bbac36-59a5-450e-a6dd-4f8918fa4815
# ╠═b2421d68-ffd0-4ea0-b4db-ac0fbcc765f4
# ╠═acac6263-20fa-406c-86c8-5bdb7e30a2c0
# ╠═2cacaf8c-0c17-442a-8c9e-999d7d9107f0
# ╟─25938291-9a12-41d2-b6c0-0f11da73fea7
# ╠═51b5c125-6e13-4f1e-a53f-73614e4e70e4
# ╠═fe90daae-fa97-4f6d-969b-35a19b9c4f4f
# ╠═3cb77ab4-10da-4028-b29a-50cf021f8e49
# ╠═328fd458-1311-42c3-a253-250d6b86c894
# ╟─607440f5-c9bf-45ef-80b9-c7f4bc065975
# ╟─9c9e402d-00ad-4708-8010-63cd6cc241ea
# ╟─783d7e9d-42f1-44e2-8631-d0ea3e3d68fb
# ╟─774c6377-97ed-4905-a2b0-334b07511935
# ╠═1492011c-d688-434e-96ff-30e071311e0a
# ╠═b986d9ea-f4e2-4254-82aa-e56a1e07053b
# ╠═ee1bec3a-ed01-4c0e-9d60-74a86d3ef1e4
# ╠═3434042a-3dc4-4940-810c-f0efb9cc9ef3
# ╠═31c499b6-dd1b-4772-b106-2f0df2ebfe41
# ╠═26048e90-3ad5-412d-bfa7-21e5085d3f43
# ╠═8957c6eb-952f-42d7-9ca1-f772e2ecf394
# ╠═1e304e6a-4248-45ee-9fa9-eaccaeaf8127
# ╠═fce06c4a-d80e-43d6-8802-76038d93267f
# ╠═54d76b66-0475-48df-8655-4c362220e1e9
# ╟─ad1760f5-422b-4f9a-bbad-2035a8902f15
# ╠═455e44c2-97b9-42ee-b2d8-8951b1991f7c
# ╠═3f93d7f0-4eec-4287-84f6-b37058c65dc8
# ╠═836f9e56-fcad-4a0e-8a7e-6b31a6a2582d
# ╠═0a7f33bf-72fb-40dd-83be-91fc83e7064f
# ╠═5b78aa8a-33ce-4ebe-a451-1fbf3cf96c1d
# ╠═bf9ac14b-3b09-40fa-a70c-e372ada461ba
# ╠═e5d07d23-aaa8-415b-9cb4-fb898c7db690
# ╟─46d24ac1-ba76-4574-ad01-052f2675794b
# ╠═e763fc92-9015-4cd1-9dbb-61e4488a1573
# ╠═f30ddb52-f36b-4051-a1ff-cc33b3fd8465
# ╠═f9d4b37b-1687-484e-8225-621fbd4b93eb
# ╠═d80aebe1-a202-4683-9507-902a01020c64
# ╠═341c57be-e3fc-4490-85f3-ccf8ca9e63ca
# ╠═92c43f2a-1acb-4f84-af1c-18a03f519eef
# ╠═86879971-6b68-4b79-84ed-0681aecc483d
# ╠═8bfc87d1-c577-435c-ad98-a2b07b52bf55
# ╠═3eaf02ff-6c9f-49c9-a421-b7803e9ccf04
# ╠═da1fe9ff-9ba4-46a4-b6fa-d2764a705767
# ╠═599f476d-a25a-4b28-8f98-75749d489333
# ╠═05e03eee-6bf0-40cf-a547-f39e8a2aba1d
# ╠═85175006-4417-441a-8b79-13a153ebf6f5
# ╠═61dbb38e-28bd-4b66-9985-da32902e2330
# ╠═61e82596-ead5-45c9-9dd7-985dec241ab7
# ╠═5252bbc4-8fc2-4226-9fa6-9f879158649f
# ╠═26e32d31-0a71-4863-9fe3-fb42ad46837b
# ╠═88e94319-09ae-4f16-98e2-70ff0f7d58a9
# ╠═e14f1d9c-5351-4fbb-b8a1-d6f24d59dd1b
# ╠═2198ef09-bd59-4357-b259-ce8d3debcfca
# ╟─302c45b9-a000-404c-837c-e3bdc062b8eb
# ╟─eefd2503-9019-4cec-916f-d104590c9de4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
