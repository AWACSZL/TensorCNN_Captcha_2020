яО
Щ¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02unknown8ьн	
Ф
sequential/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namesequential/conv2d/kernel
Н
,sequential/conv2d/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d/kernel*&
_output_shapes
: *
dtype0
Д
sequential/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namesequential/conv2d/bias
}
*sequential/conv2d/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d/bias*
_output_shapes
: *
dtype0
С
sequential/p_re_lu/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape::Ю *)
shared_namesequential/p_re_lu/alpha
К
,sequential/p_re_lu/alpha/Read/ReadVariableOpReadVariableOpsequential/p_re_lu/alpha*#
_output_shapes
::Ю *
dtype0
Ш
sequential/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_namesequential/conv2d_1/kernel
С
.sequential/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
И
sequential/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namesequential/conv2d_1/bias
Б
,sequential/conv2d_1/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d_1/bias*
_output_shapes
:@*
dtype0
Ф
sequential/p_re_lu_1/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:K@*+
shared_namesequential/p_re_lu_1/alpha
Н
.sequential/p_re_lu_1/alpha/Read/ReadVariableOpReadVariableOpsequential/p_re_lu_1/alpha*"
_output_shapes
:K@*
dtype0
Щ
sequential/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*+
shared_namesequential/conv2d_2/kernel
Т
.sequential/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d_2/kernel*'
_output_shapes
:@А*
dtype0
Й
sequential/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_namesequential/conv2d_2/bias
В
,sequential/conv2d_2/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d_2/bias*
_output_shapes	
:А*
dtype0
Х
sequential/p_re_lu_2/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:!А*+
shared_namesequential/p_re_lu_2/alpha
О
.sequential/p_re_lu_2/alpha/Read/ReadVariableOpReadVariableOpsequential/p_re_lu_2/alpha*#
_output_shapes
:!А*
dtype0
Л
sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@(*(
shared_namesequential/dense/kernel
Д
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel*
_output_shapes
:	А@(*
dtype0
В
sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_namesequential/dense/bias
{
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes
:(*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
в
Adam/sequential/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/sequential/conv2d/kernel/m
Ы
3Adam/sequential/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/kernel/m*&
_output_shapes
: *
dtype0
Т
Adam/sequential/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/sequential/conv2d/bias/m
Л
1Adam/sequential/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/bias/m*
_output_shapes
: *
dtype0
Я
Adam/sequential/p_re_lu/alpha/mVarHandleOp*
_output_shapes
: *
dtype0*
shape::Ю *0
shared_name!Adam/sequential/p_re_lu/alpha/m
Ш
3Adam/sequential/p_re_lu/alpha/m/Read/ReadVariableOpReadVariableOpAdam/sequential/p_re_lu/alpha/m*#
_output_shapes
::Ю *
dtype0
ж
!Adam/sequential/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/sequential/conv2d_1/kernel/m
Я
5Adam/sequential/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0
Ц
Adam/sequential/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/sequential/conv2d_1/bias/m
П
3Adam/sequential/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
в
!Adam/sequential/p_re_lu_1/alpha/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:K@*2
shared_name#!Adam/sequential/p_re_lu_1/alpha/m
Ы
5Adam/sequential/p_re_lu_1/alpha/m/Read/ReadVariableOpReadVariableOp!Adam/sequential/p_re_lu_1/alpha/m*"
_output_shapes
:K@*
dtype0
з
!Adam/sequential/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*2
shared_name#!Adam/sequential/conv2d_2/kernel/m
а
5Adam/sequential/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_2/kernel/m*'
_output_shapes
:@А*
dtype0
Ч
Adam/sequential/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/sequential/conv2d_2/bias/m
Р
3Adam/sequential/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_2/bias/m*
_output_shapes	
:А*
dtype0
г
!Adam/sequential/p_re_lu_2/alpha/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:!А*2
shared_name#!Adam/sequential/p_re_lu_2/alpha/m
Ь
5Adam/sequential/p_re_lu_2/alpha/m/Read/ReadVariableOpReadVariableOp!Adam/sequential/p_re_lu_2/alpha/m*#
_output_shapes
:!А*
dtype0
Щ
Adam/sequential/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@(*/
shared_name Adam/sequential/dense/kernel/m
Т
2Adam/sequential/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/kernel/m*
_output_shapes
:	А@(*
dtype0
Р
Adam/sequential/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*-
shared_nameAdam/sequential/dense/bias/m
Й
0Adam/sequential/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/bias/m*
_output_shapes
:(*
dtype0
в
Adam/sequential/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/sequential/conv2d/kernel/v
Ы
3Adam/sequential/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/kernel/v*&
_output_shapes
: *
dtype0
Т
Adam/sequential/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/sequential/conv2d/bias/v
Л
1Adam/sequential/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/bias/v*
_output_shapes
: *
dtype0
Я
Adam/sequential/p_re_lu/alpha/vVarHandleOp*
_output_shapes
: *
dtype0*
shape::Ю *0
shared_name!Adam/sequential/p_re_lu/alpha/v
Ш
3Adam/sequential/p_re_lu/alpha/v/Read/ReadVariableOpReadVariableOpAdam/sequential/p_re_lu/alpha/v*#
_output_shapes
::Ю *
dtype0
ж
!Adam/sequential/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/sequential/conv2d_1/kernel/v
Я
5Adam/sequential/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0
Ц
Adam/sequential/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/sequential/conv2d_1/bias/v
П
3Adam/sequential/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
в
!Adam/sequential/p_re_lu_1/alpha/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:K@*2
shared_name#!Adam/sequential/p_re_lu_1/alpha/v
Ы
5Adam/sequential/p_re_lu_1/alpha/v/Read/ReadVariableOpReadVariableOp!Adam/sequential/p_re_lu_1/alpha/v*"
_output_shapes
:K@*
dtype0
з
!Adam/sequential/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*2
shared_name#!Adam/sequential/conv2d_2/kernel/v
а
5Adam/sequential/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_2/kernel/v*'
_output_shapes
:@А*
dtype0
Ч
Adam/sequential/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/sequential/conv2d_2/bias/v
Р
3Adam/sequential/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_2/bias/v*
_output_shapes	
:А*
dtype0
г
!Adam/sequential/p_re_lu_2/alpha/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:!А*2
shared_name#!Adam/sequential/p_re_lu_2/alpha/v
Ь
5Adam/sequential/p_re_lu_2/alpha/v/Read/ReadVariableOpReadVariableOp!Adam/sequential/p_re_lu_2/alpha/v*#
_output_shapes
:!А*
dtype0
Щ
Adam/sequential/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@(*/
shared_name Adam/sequential/dense/kernel/v
Т
2Adam/sequential/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/kernel/v*
_output_shapes
:	А@(*
dtype0
Р
Adam/sequential/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*-
shared_nameAdam/sequential/dense/bias/v
Й
0Adam/sequential/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/bias/v*
_output_shapes
:(*
dtype0

NoOpNoOp
╚H
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ГH
value∙GBЎG BяG
Э
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
]
	alpha
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
 trainable_variables
!	variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
]
	)alpha
*regularization_losses
+trainable_variables
,	variables
-	keras_api
R
.regularization_losses
/trainable_variables
0	variables
1	keras_api
h

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
]
	8alpha
9regularization_losses
:trainable_variables
;	variables
<	keras_api
R
=regularization_losses
>trainable_variables
?	variables
@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
h

Ekernel
Fbias
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
R
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
R
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
Ь
Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_ratemЬmЭmЮ#mЯ$mа)mб2mв3mг8mдEmеFmжvзvиvй#vк$vл)vм2vн3vо8vпEv░Fv▒
 
N
0
1
2
#3
$4
)5
26
37
88
E9
F10
N
0
1
2
#3
$4
)5
26
37
88
E9
F10
Ъ
Xmetrics
regularization_losses
Ylayer_regularization_losses
Znon_trainable_variables
trainable_variables

[layers
	variables
 
WU
VARIABLE_VALUEsequential/conv2d/kernel)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEsequential/conv2d/bias'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Ъ
\metrics
regularization_losses
]layer_regularization_losses
trainable_variables
^non_trainable_variables

_layers
	variables
VT
VARIABLE_VALUEsequential/p_re_lu/alpha(layer-1/alpha/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
Ъ
`metrics
regularization_losses
alayer_regularization_losses
trainable_variables
bnon_trainable_variables

clayers
	variables
 
 
 
Ъ
dmetrics
regularization_losses
elayer_regularization_losses
 trainable_variables
fnon_trainable_variables

glayers
!	variables
YW
VARIABLE_VALUEsequential/conv2d_1/kernel)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEsequential/conv2d_1/bias'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
Ъ
hmetrics
%regularization_losses
ilayer_regularization_losses
&trainable_variables
jnon_trainable_variables

klayers
'	variables
XV
VARIABLE_VALUEsequential/p_re_lu_1/alpha(layer-4/alpha/.ATTRIBUTES/VARIABLE_VALUE
 

)0

)0
Ъ
lmetrics
*regularization_losses
mlayer_regularization_losses
+trainable_variables
nnon_trainable_variables

olayers
,	variables
 
 
 
Ъ
pmetrics
.regularization_losses
qlayer_regularization_losses
/trainable_variables
rnon_trainable_variables

slayers
0	variables
YW
VARIABLE_VALUEsequential/conv2d_2/kernel)layer-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEsequential/conv2d_2/bias'layer-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

20
31

20
31
Ъ
tmetrics
4regularization_losses
ulayer_regularization_losses
5trainable_variables
vnon_trainable_variables

wlayers
6	variables
XV
VARIABLE_VALUEsequential/p_re_lu_2/alpha(layer-7/alpha/.ATTRIBUTES/VARIABLE_VALUE
 

80

80
Ъ
xmetrics
9regularization_losses
ylayer_regularization_losses
:trainable_variables
znon_trainable_variables

{layers
;	variables
 
 
 
Ъ
|metrics
=regularization_losses
}layer_regularization_losses
>trainable_variables
~non_trainable_variables

layers
?	variables
 
 
 
Ю
Аmetrics
Aregularization_losses
 Бlayer_regularization_losses
Btrainable_variables
Вnon_trainable_variables
Гlayers
C	variables
WU
VARIABLE_VALUEsequential/dense/kernel*layer-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEsequential/dense/bias(layer-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
Ю
Дmetrics
Gregularization_losses
 Еlayer_regularization_losses
Htrainable_variables
Жnon_trainable_variables
Зlayers
I	variables
 
 
 
Ю
Иmetrics
Kregularization_losses
 Йlayer_regularization_losses
Ltrainable_variables
Кnon_trainable_variables
Лlayers
M	variables
 
 
 
Ю
Мmetrics
Oregularization_losses
 Нlayer_regularization_losses
Ptrainable_variables
Оnon_trainable_variables
Пlayers
Q	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

Р0
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


Сtotal

Тcount
У
_fn_kwargs
Фregularization_losses
Хtrainable_variables
Ц	variables
Ч	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

С0
Т1
б
Шmetrics
Фregularization_losses
 Щlayer_regularization_losses
Хtrainable_variables
Ъnon_trainable_variables
Ыlayers
Ц	variables
 
 

С0
Т1
 
zx
VARIABLE_VALUEAdam/sequential/conv2d/kernel/mElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/sequential/conv2d/bias/mClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/sequential/p_re_lu/alpha/mDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE!Adam/sequential/conv2d_1/kernel/mElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/sequential/conv2d_1/bias/mClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/sequential/p_re_lu_1/alpha/mDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE!Adam/sequential/conv2d_2/kernel/mElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/sequential/conv2d_2/bias/mClayer-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/sequential/p_re_lu_2/alpha/mDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/sequential/dense/kernel/mFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/sequential/dense/bias/mDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/sequential/conv2d/kernel/vElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/sequential/conv2d/bias/vClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/sequential/p_re_lu/alpha/vDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE!Adam/sequential/conv2d_1/kernel/vElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/sequential/conv2d_1/bias/vClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/sequential/p_re_lu_1/alpha/vDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE!Adam/sequential/conv2d_2/kernel/vElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/sequential/conv2d_2/bias/vClayer-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/sequential/p_re_lu_2/alpha/vDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/sequential/dense/kernel/vFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/sequential/dense/bias/vDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
М
serving_default_input_1Placeholder*0
_output_shapes
:         <а*
dtype0*%
shape:         <а
╔
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential/conv2d/kernelsequential/conv2d/biassequential/p_re_lu/alphasequential/conv2d_1/kernelsequential/conv2d_1/biassequential/p_re_lu_1/alphasequential/conv2d_2/kernelsequential/conv2d_2/biassequential/p_re_lu_2/alphasequential/dense/kernelsequential/dense/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference_signature_wrapper_866151
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ъ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,sequential/conv2d/kernel/Read/ReadVariableOp*sequential/conv2d/bias/Read/ReadVariableOp,sequential/p_re_lu/alpha/Read/ReadVariableOp.sequential/conv2d_1/kernel/Read/ReadVariableOp,sequential/conv2d_1/bias/Read/ReadVariableOp.sequential/p_re_lu_1/alpha/Read/ReadVariableOp.sequential/conv2d_2/kernel/Read/ReadVariableOp,sequential/conv2d_2/bias/Read/ReadVariableOp.sequential/p_re_lu_2/alpha/Read/ReadVariableOp+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3Adam/sequential/conv2d/kernel/m/Read/ReadVariableOp1Adam/sequential/conv2d/bias/m/Read/ReadVariableOp3Adam/sequential/p_re_lu/alpha/m/Read/ReadVariableOp5Adam/sequential/conv2d_1/kernel/m/Read/ReadVariableOp3Adam/sequential/conv2d_1/bias/m/Read/ReadVariableOp5Adam/sequential/p_re_lu_1/alpha/m/Read/ReadVariableOp5Adam/sequential/conv2d_2/kernel/m/Read/ReadVariableOp3Adam/sequential/conv2d_2/bias/m/Read/ReadVariableOp5Adam/sequential/p_re_lu_2/alpha/m/Read/ReadVariableOp2Adam/sequential/dense/kernel/m/Read/ReadVariableOp0Adam/sequential/dense/bias/m/Read/ReadVariableOp3Adam/sequential/conv2d/kernel/v/Read/ReadVariableOp1Adam/sequential/conv2d/bias/v/Read/ReadVariableOp3Adam/sequential/p_re_lu/alpha/v/Read/ReadVariableOp5Adam/sequential/conv2d_1/kernel/v/Read/ReadVariableOp3Adam/sequential/conv2d_1/bias/v/Read/ReadVariableOp5Adam/sequential/p_re_lu_1/alpha/v/Read/ReadVariableOp5Adam/sequential/conv2d_2/kernel/v/Read/ReadVariableOp3Adam/sequential/conv2d_2/bias/v/Read/ReadVariableOp5Adam/sequential/p_re_lu_2/alpha/v/Read/ReadVariableOp2Adam/sequential/dense/kernel/v/Read/ReadVariableOp0Adam/sequential/dense/bias/v/Read/ReadVariableOpConst*5
Tin.
,2*	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_save_866517
ї

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential/conv2d/kernelsequential/conv2d/biassequential/p_re_lu/alphasequential/conv2d_1/kernelsequential/conv2d_1/biassequential/p_re_lu_1/alphasequential/conv2d_2/kernelsequential/conv2d_2/biassequential/p_re_lu_2/alphasequential/dense/kernelsequential/dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/sequential/conv2d/kernel/mAdam/sequential/conv2d/bias/mAdam/sequential/p_re_lu/alpha/m!Adam/sequential/conv2d_1/kernel/mAdam/sequential/conv2d_1/bias/m!Adam/sequential/p_re_lu_1/alpha/m!Adam/sequential/conv2d_2/kernel/mAdam/sequential/conv2d_2/bias/m!Adam/sequential/p_re_lu_2/alpha/mAdam/sequential/dense/kernel/mAdam/sequential/dense/bias/mAdam/sequential/conv2d/kernel/vAdam/sequential/conv2d/bias/vAdam/sequential/p_re_lu/alpha/v!Adam/sequential/conv2d_1/kernel/vAdam/sequential/conv2d_1/bias/v!Adam/sequential/p_re_lu_1/alpha/v!Adam/sequential/conv2d_2/kernel/vAdam/sequential/conv2d_2/bias/v!Adam/sequential/p_re_lu_2/alpha/vAdam/sequential/dense/kernel/vAdam/sequential/dense/bias/v*4
Tin-
+2)*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__traced_restore_866649ещ
╝
и
'__inference_conv2d_layer_call_fn_865780

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                            **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_8657722
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
иж
╚
"__inference__traced_restore_866649
file_prefix-
)assignvariableop_sequential_conv2d_kernel-
)assignvariableop_1_sequential_conv2d_bias/
+assignvariableop_2_sequential_p_re_lu_alpha1
-assignvariableop_3_sequential_conv2d_1_kernel/
+assignvariableop_4_sequential_conv2d_1_bias1
-assignvariableop_5_sequential_p_re_lu_1_alpha1
-assignvariableop_6_sequential_conv2d_2_kernel/
+assignvariableop_7_sequential_conv2d_2_bias1
-assignvariableop_8_sequential_p_re_lu_2_alpha.
*assignvariableop_9_sequential_dense_kernel-
)assignvariableop_10_sequential_dense_bias!
assignvariableop_11_adam_iter#
assignvariableop_12_adam_beta_1#
assignvariableop_13_adam_beta_2"
assignvariableop_14_adam_decay*
&assignvariableop_15_adam_learning_rate
assignvariableop_16_total
assignvariableop_17_count7
3assignvariableop_18_adam_sequential_conv2d_kernel_m5
1assignvariableop_19_adam_sequential_conv2d_bias_m7
3assignvariableop_20_adam_sequential_p_re_lu_alpha_m9
5assignvariableop_21_adam_sequential_conv2d_1_kernel_m7
3assignvariableop_22_adam_sequential_conv2d_1_bias_m9
5assignvariableop_23_adam_sequential_p_re_lu_1_alpha_m9
5assignvariableop_24_adam_sequential_conv2d_2_kernel_m7
3assignvariableop_25_adam_sequential_conv2d_2_bias_m9
5assignvariableop_26_adam_sequential_p_re_lu_2_alpha_m6
2assignvariableop_27_adam_sequential_dense_kernel_m4
0assignvariableop_28_adam_sequential_dense_bias_m7
3assignvariableop_29_adam_sequential_conv2d_kernel_v5
1assignvariableop_30_adam_sequential_conv2d_bias_v7
3assignvariableop_31_adam_sequential_p_re_lu_alpha_v9
5assignvariableop_32_adam_sequential_conv2d_1_kernel_v7
3assignvariableop_33_adam_sequential_conv2d_1_bias_v9
5assignvariableop_34_adam_sequential_p_re_lu_1_alpha_v9
5assignvariableop_35_adam_sequential_conv2d_2_kernel_v7
3assignvariableop_36_adam_sequential_conv2d_2_bias_v9
5assignvariableop_37_adam_sequential_p_re_lu_2_alpha_v6
2assignvariableop_38_adam_sequential_dense_kernel_v4
0assignvariableop_39_adam_sequential_dense_bias_v
identity_41ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1▓
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*╛
value┤B▒(B)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-1/alpha/.ATTRIBUTES/VARIABLE_VALUEB)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-4/alpha/.ATTRIBUTES/VARIABLE_VALUEB)layer-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-6/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-7/alpha/.ATTRIBUTES/VARIABLE_VALUEB*layer-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB(layer-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names▐
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╢
_output_shapesг
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЩ
AssignVariableOpAssignVariableOp)assignvariableop_sequential_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Я
AssignVariableOp_1AssignVariableOp)assignvariableop_1_sequential_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2б
AssignVariableOp_2AssignVariableOp+assignvariableop_2_sequential_p_re_lu_alphaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOp-assignvariableop_3_sequential_conv2d_1_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4б
AssignVariableOp_4AssignVariableOp+assignvariableop_4_sequential_conv2d_1_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5г
AssignVariableOp_5AssignVariableOp-assignvariableop_5_sequential_p_re_lu_1_alphaIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6г
AssignVariableOp_6AssignVariableOp-assignvariableop_6_sequential_conv2d_2_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7б
AssignVariableOp_7AssignVariableOp+assignvariableop_7_sequential_conv2d_2_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8г
AssignVariableOp_8AssignVariableOp-assignvariableop_8_sequential_p_re_lu_2_alphaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9а
AssignVariableOp_9AssignVariableOp*assignvariableop_9_sequential_dense_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10в
AssignVariableOp_10AssignVariableOp)assignvariableop_10_sequential_dense_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0	*
_output_shapes
:2
Identity_11Ц
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ш
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ш
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Ч
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Я
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16Т
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Т
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18м
AssignVariableOp_18AssignVariableOp3assignvariableop_18_adam_sequential_conv2d_kernel_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19к
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_sequential_conv2d_bias_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20м
AssignVariableOp_20AssignVariableOp3assignvariableop_20_adam_sequential_p_re_lu_alpha_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21о
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_sequential_conv2d_1_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22м
AssignVariableOp_22AssignVariableOp3assignvariableop_22_adam_sequential_conv2d_1_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23о
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_sequential_p_re_lu_1_alpha_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24о
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_sequential_conv2d_2_kernel_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25м
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adam_sequential_conv2d_2_bias_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26о
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_sequential_p_re_lu_2_alpha_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27л
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_sequential_dense_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28й
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_sequential_dense_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29м
AssignVariableOp_29AssignVariableOp3assignvariableop_29_adam_sequential_conv2d_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30к
AssignVariableOp_30AssignVariableOp1assignvariableop_30_adam_sequential_conv2d_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31м
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adam_sequential_p_re_lu_alpha_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32о
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_sequential_conv2d_1_kernel_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33м
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_sequential_conv2d_1_bias_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34о
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_sequential_p_re_lu_1_alpha_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35о
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_sequential_conv2d_2_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36м
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_sequential_conv2d_2_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37о
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_sequential_p_re_lu_2_alpha_vIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38л
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_sequential_dense_kernel_vIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39й
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_sequential_dense_bias_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╬
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_40█
Identity_41IdentityIdentity_40:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_41"#
identity_41Identity_41:output:0*╖
_input_shapesе
в: ::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
▌
D
(__inference_softmax_layer_call_fn_866373

inputs
identityп
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_8660002
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0**
_input_shapes
:         
:& "
 
_user_specified_nameinputs
э
з
&__inference_dense_layer_call_fn_866345

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         (**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_8659622
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         (2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ю7
╝
F__inference_sequential_layer_call_and_return_conditional_losses_866068

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2*
&p_re_lu_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2,
(p_re_lu_1_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2,
(p_re_lu_2_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвp_re_lu/StatefulPartitionedCallв!p_re_lu_1/StatefulPartitionedCallв!p_re_lu_2/StatefulPartitionedCallй
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         :Ю **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_8657722 
conv2d/StatefulPartitionedCallж
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0&p_re_lu_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         :Ю **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_p_re_lu_layer_call_and_return_conditional_losses_8657932!
p_re_lu/StatefulPartitionedCallў
max_pooling2d/PartitionedCallPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         O **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_8658062
max_pooling2d/PartitionedCall╥
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         K@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_8658242"
 conv2d_1/StatefulPartitionedCallп
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0(p_re_lu_1_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         K@**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_p_re_lu_1_layer_call_and_return_conditional_losses_8658452#
!p_re_lu_1/StatefulPartitionedCall 
max_pooling2d_1/PartitionedCallPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         %@**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8658582!
max_pooling2d_1/PartitionedCall╒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         !А**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_8658762"
 conv2d_2/StatefulPartitionedCall░
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0(p_re_lu_2_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         !А**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_p_re_lu_2_layer_call_and_return_conditional_losses_8658972#
!p_re_lu_2/StatefulPartitionedCallА
max_pooling2d_2/PartitionedCallPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8659102!
max_pooling2d_2/PartitionedCall▐
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8659442
flatten/PartitionedCall╡
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         (**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_8659622
dense/StatefulPartitionedCall▀
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_8659872
reshape/PartitionedCall┘
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_8660002
softmax/PartitionedCallщ
IdentityIdentity softmax/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
│
_
C__inference_reshape_layer_call_and_return_conditional_losses_866358

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         
2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*&
_input_shapes
:         (:& "
 
_user_specified_nameinputs
Ь
_
C__inference_softmax_layer_call_and_return_conditional_losses_866368

inputs
identity[
SoftmaxSoftmaxinputs*
T0*+
_output_shapes
:         
2	
Softmaxi
IdentityIdentitySoftmax:softmax:0*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0**
_input_shapes
:         
:& "
 
_user_specified_nameinputs
▄
D
(__inference_flatten_layer_call_fn_866328

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8659442
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╦
L
0__inference_max_pooling2d_2_layer_call_fn_865916

inputs
identity╓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8659102
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
є

▌
D__inference_conv2d_1_layer_call_and_return_conditional_losses_865824

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddп
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
█O
Л
F__inference_sequential_layer_call_and_return_conditional_losses_866285

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource#
p_re_lu_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource%
!p_re_lu_1_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource%
!p_re_lu_2_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвp_re_lu/ReadVariableOpвp_re_lu_1/ReadVariableOpвp_re_lu_2/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp║
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         :Ю *
paddingVALID*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpе
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         :Ю 2
conv2d/BiasAddx
p_re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:         :Ю 2
p_re_lu/ReluХ
p_re_lu/ReadVariableOpReadVariableOpp_re_lu_readvariableop_resource*#
_output_shapes
::Ю *
dtype02
p_re_lu/ReadVariableOpo
p_re_lu/NegNegp_re_lu/ReadVariableOp:value:0*
T0*#
_output_shapes
::Ю 2
p_re_lu/Negy
p_re_lu/Neg_1Negconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:         :Ю 2
p_re_lu/Neg_1v
p_re_lu/Relu_1Relup_re_lu/Neg_1:y:0*
T0*0
_output_shapes
:         :Ю 2
p_re_lu/Relu_1Л
p_re_lu/mulMulp_re_lu/Neg:y:0p_re_lu/Relu_1:activations:0*
T0*0
_output_shapes
:         :Ю 2
p_re_lu/mulЛ
p_re_lu/addAddV2p_re_lu/Relu:activations:0p_re_lu/mul:z:0*
T0*0
_output_shapes
:         :Ю 2
p_re_lu/add╖
max_pooling2d/MaxPoolMaxPoolp_re_lu/add:z:0*/
_output_shapes
:         O *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╫
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         K@*
paddingVALID*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         K@2
conv2d_1/BiasAdd}
p_re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         K@2
p_re_lu_1/ReluЪ
p_re_lu_1/ReadVariableOpReadVariableOp!p_re_lu_1_readvariableop_resource*"
_output_shapes
:K@*
dtype02
p_re_lu_1/ReadVariableOpt
p_re_lu_1/NegNeg p_re_lu_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:K@2
p_re_lu_1/Neg~
p_re_lu_1/Neg_1Negconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         K@2
p_re_lu_1/Neg_1{
p_re_lu_1/Relu_1Relup_re_lu_1/Neg_1:y:0*
T0*/
_output_shapes
:         K@2
p_re_lu_1/Relu_1Т
p_re_lu_1/mulMulp_re_lu_1/Neg:y:0p_re_lu_1/Relu_1:activations:0*
T0*/
_output_shapes
:         K@2
p_re_lu_1/mulТ
p_re_lu_1/addAddV2p_re_lu_1/Relu:activations:0p_re_lu_1/mul:z:0*
T0*/
_output_shapes
:         K@2
p_re_lu_1/add╜
max_pooling2d_1/MaxPoolMaxPoolp_re_lu_1/add:z:0*/
_output_shapes
:         %@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool▒
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_2/Conv2D/ReadVariableOp┌
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         !А*
paddingVALID*
strides
2
conv2d_2/Conv2Dи
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpн
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         !А2
conv2d_2/BiasAdd~
p_re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         !А2
p_re_lu_2/ReluЫ
p_re_lu_2/ReadVariableOpReadVariableOp!p_re_lu_2_readvariableop_resource*#
_output_shapes
:!А*
dtype02
p_re_lu_2/ReadVariableOpu
p_re_lu_2/NegNeg p_re_lu_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:!А2
p_re_lu_2/Neg
p_re_lu_2/Neg_1Negconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         !А2
p_re_lu_2/Neg_1|
p_re_lu_2/Relu_1Relup_re_lu_2/Neg_1:y:0*
T0*0
_output_shapes
:         !А2
p_re_lu_2/Relu_1У
p_re_lu_2/mulMulp_re_lu_2/Neg:y:0p_re_lu_2/Relu_1:activations:0*
T0*0
_output_shapes
:         !А2
p_re_lu_2/mulУ
p_re_lu_2/addAddV2p_re_lu_2/Relu:activations:0p_re_lu_2/mul:z:0*
T0*0
_output_shapes
:         !А2
p_re_lu_2/add╛
max_pooling2d_2/MaxPoolMaxPoolp_re_lu_2/add:z:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten/ConstЪ
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten/Reshapeа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А@(*
dtype02
dense/MatMul/ReadVariableOpЧ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (2
dense/BiasAddd
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape/Reshape/shape/2╚
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЫ
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         
2
reshape/Reshape}
softmax/SoftmaxSoftmaxreshape/Reshape:output:0*
T0*+
_output_shapes
:         
2
softmax/Softmax┬
IdentityIdentitysoftmax/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^p_re_lu/ReadVariableOp^p_re_lu_1/ReadVariableOp^p_re_lu_2/ReadVariableOp*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp20
p_re_lu/ReadVariableOpp_re_lu/ReadVariableOp24
p_re_lu_1/ReadVariableOpp_re_lu_1/ReadVariableOp24
p_re_lu_2/ReadVariableOpp_re_lu_2/ReadVariableOp:& "
 
_user_specified_nameinputs
Л
_
C__inference_flatten_layer_call_and_return_conditional_losses_865944

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Ь
_
C__inference_softmax_layer_call_and_return_conditional_losses_866000

inputs
identity[
SoftmaxSoftmaxinputs*
T0*+
_output_shapes
:         
2	
Softmaxi
IdentityIdentitySoftmax:softmax:0*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0**
_input_shapes
:         
:& "
 
_user_specified_nameinputs
ё7
╜
F__inference_sequential_layer_call_and_return_conditional_losses_866009
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2*
&p_re_lu_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2,
(p_re_lu_1_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2,
(p_re_lu_2_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвp_re_lu/StatefulPartitionedCallв!p_re_lu_1/StatefulPartitionedCallв!p_re_lu_2/StatefulPartitionedCallк
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         :Ю **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_8657722 
conv2d/StatefulPartitionedCallж
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0&p_re_lu_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         :Ю **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_p_re_lu_layer_call_and_return_conditional_losses_8657932!
p_re_lu/StatefulPartitionedCallў
max_pooling2d/PartitionedCallPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         O **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_8658062
max_pooling2d/PartitionedCall╥
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         K@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_8658242"
 conv2d_1/StatefulPartitionedCallп
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0(p_re_lu_1_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         K@**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_p_re_lu_1_layer_call_and_return_conditional_losses_8658452#
!p_re_lu_1/StatefulPartitionedCall 
max_pooling2d_1/PartitionedCallPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         %@**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8658582!
max_pooling2d_1/PartitionedCall╒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         !А**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_8658762"
 conv2d_2/StatefulPartitionedCall░
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0(p_re_lu_2_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         !А**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_p_re_lu_2_layer_call_and_return_conditional_losses_8658972#
!p_re_lu_2/StatefulPartitionedCallА
max_pooling2d_2/PartitionedCallPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8659102!
max_pooling2d_2/PartitionedCall▐
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8659442
flatten/PartitionedCall╡
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         (**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_8659622
dense/StatefulPartitionedCall▀
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_8659872
reshape/PartitionedCall┘
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_8660002
softmax/PartitionedCallщ
IdentityIdentity softmax/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
ю7
╝
F__inference_sequential_layer_call_and_return_conditional_losses_866112

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2*
&p_re_lu_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2,
(p_re_lu_1_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2,
(p_re_lu_2_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвp_re_lu/StatefulPartitionedCallв!p_re_lu_1/StatefulPartitionedCallв!p_re_lu_2/StatefulPartitionedCallй
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         :Ю **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_8657722 
conv2d/StatefulPartitionedCallж
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0&p_re_lu_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         :Ю **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_p_re_lu_layer_call_and_return_conditional_losses_8657932!
p_re_lu/StatefulPartitionedCallў
max_pooling2d/PartitionedCallPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         O **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_8658062
max_pooling2d/PartitionedCall╥
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         K@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_8658242"
 conv2d_1/StatefulPartitionedCallп
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0(p_re_lu_1_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         K@**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_p_re_lu_1_layer_call_and_return_conditional_losses_8658452#
!p_re_lu_1/StatefulPartitionedCall 
max_pooling2d_1/PartitionedCallPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         %@**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8658582!
max_pooling2d_1/PartitionedCall╒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         !А**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_8658762"
 conv2d_2/StatefulPartitionedCall░
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0(p_re_lu_2_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         !А**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_p_re_lu_2_layer_call_and_return_conditional_losses_8658972#
!p_re_lu_2/StatefulPartitionedCallА
max_pooling2d_2/PartitionedCallPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8659102!
max_pooling2d_2/PartitionedCall▐
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8659442
flatten/PartitionedCall╡
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         (**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_8659622
dense/StatefulPartitionedCall▀
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_8659872
reshape/PartitionedCall┘
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_8660002
softmax/PartitionedCallщ
IdentityIdentity softmax/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
╡
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_865910

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
╒	
Р
C__inference_p_re_lu_layer_call_and_return_conditional_losses_865793

inputs
readvariableop_resource
identityИвReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4                                    2
Relu}
ReadVariableOpReadVariableOpreadvariableop_resource*#
_output_shapes
::Ю *
dtype02
ReadVariableOpW
NegNegReadVariableOp:value:0*
T0*#
_output_shapes
::Ю 2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4                                    2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4                                    2
Relu_1k
mulMulNeg:y:0Relu_1:activations:0*
T0*0
_output_shapes
:         :Ю 2
mulk
addAddV2Relu:activations:0mul:z:0*
T0*0
_output_shapes
:         :Ю 2
addu
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:         :Ю 2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4                                    :2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs
╦
L
0__inference_max_pooling2d_1_layer_call_fn_865864

inputs
identity╓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8658582
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
╫	
Т
E__inference_p_re_lu_2_layer_call_and_return_conditional_losses_865897

inputs
readvariableop_resource
identityИвReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4                                    2
Relu}
ReadVariableOpReadVariableOpreadvariableop_resource*#
_output_shapes
:!А*
dtype02
ReadVariableOpW
NegNegReadVariableOp:value:0*
T0*#
_output_shapes
:!А2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4                                    2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4                                    2
Relu_1k
mulMulNeg:y:0Relu_1:activations:0*
T0*0
_output_shapes
:         !А2
mulk
addAddV2Relu:activations:0mul:z:0*
T0*0
_output_shapes
:         !А2
addu
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:         !А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4                                    :2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs
█O
Л
F__inference_sequential_layer_call_and_return_conditional_losses_866218

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource#
p_re_lu_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource%
!p_re_lu_1_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource%
!p_re_lu_2_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвp_re_lu/ReadVariableOpвp_re_lu_1/ReadVariableOpвp_re_lu_2/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp║
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         :Ю *
paddingVALID*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpе
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         :Ю 2
conv2d/BiasAddx
p_re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:         :Ю 2
p_re_lu/ReluХ
p_re_lu/ReadVariableOpReadVariableOpp_re_lu_readvariableop_resource*#
_output_shapes
::Ю *
dtype02
p_re_lu/ReadVariableOpo
p_re_lu/NegNegp_re_lu/ReadVariableOp:value:0*
T0*#
_output_shapes
::Ю 2
p_re_lu/Negy
p_re_lu/Neg_1Negconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:         :Ю 2
p_re_lu/Neg_1v
p_re_lu/Relu_1Relup_re_lu/Neg_1:y:0*
T0*0
_output_shapes
:         :Ю 2
p_re_lu/Relu_1Л
p_re_lu/mulMulp_re_lu/Neg:y:0p_re_lu/Relu_1:activations:0*
T0*0
_output_shapes
:         :Ю 2
p_re_lu/mulЛ
p_re_lu/addAddV2p_re_lu/Relu:activations:0p_re_lu/mul:z:0*
T0*0
_output_shapes
:         :Ю 2
p_re_lu/add╖
max_pooling2d/MaxPoolMaxPoolp_re_lu/add:z:0*/
_output_shapes
:         O *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╫
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         K@*
paddingVALID*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         K@2
conv2d_1/BiasAdd}
p_re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         K@2
p_re_lu_1/ReluЪ
p_re_lu_1/ReadVariableOpReadVariableOp!p_re_lu_1_readvariableop_resource*"
_output_shapes
:K@*
dtype02
p_re_lu_1/ReadVariableOpt
p_re_lu_1/NegNeg p_re_lu_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:K@2
p_re_lu_1/Neg~
p_re_lu_1/Neg_1Negconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         K@2
p_re_lu_1/Neg_1{
p_re_lu_1/Relu_1Relup_re_lu_1/Neg_1:y:0*
T0*/
_output_shapes
:         K@2
p_re_lu_1/Relu_1Т
p_re_lu_1/mulMulp_re_lu_1/Neg:y:0p_re_lu_1/Relu_1:activations:0*
T0*/
_output_shapes
:         K@2
p_re_lu_1/mulТ
p_re_lu_1/addAddV2p_re_lu_1/Relu:activations:0p_re_lu_1/mul:z:0*
T0*/
_output_shapes
:         K@2
p_re_lu_1/add╜
max_pooling2d_1/MaxPoolMaxPoolp_re_lu_1/add:z:0*/
_output_shapes
:         %@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool▒
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_2/Conv2D/ReadVariableOp┌
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         !А*
paddingVALID*
strides
2
conv2d_2/Conv2Dи
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpн
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         !А2
conv2d_2/BiasAdd~
p_re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         !А2
p_re_lu_2/ReluЫ
p_re_lu_2/ReadVariableOpReadVariableOp!p_re_lu_2_readvariableop_resource*#
_output_shapes
:!А*
dtype02
p_re_lu_2/ReadVariableOpu
p_re_lu_2/NegNeg p_re_lu_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:!А2
p_re_lu_2/Neg
p_re_lu_2/Neg_1Negconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         !А2
p_re_lu_2/Neg_1|
p_re_lu_2/Relu_1Relup_re_lu_2/Neg_1:y:0*
T0*0
_output_shapes
:         !А2
p_re_lu_2/Relu_1У
p_re_lu_2/mulMulp_re_lu_2/Neg:y:0p_re_lu_2/Relu_1:activations:0*
T0*0
_output_shapes
:         !А2
p_re_lu_2/mulУ
p_re_lu_2/addAddV2p_re_lu_2/Relu:activations:0p_re_lu_2/mul:z:0*
T0*0
_output_shapes
:         !А2
p_re_lu_2/add╛
max_pooling2d_2/MaxPoolMaxPoolp_re_lu_2/add:z:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten/ConstЪ
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten/Reshapeа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А@(*
dtype02
dense/MatMul/ReadVariableOpЧ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (2
dense/BiasAddd
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape/Reshape/shape/2╚
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЫ
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         
2
reshape/Reshape}
softmax/SoftmaxSoftmaxreshape/Reshape:output:0*
T0*+
_output_shapes
:         
2
softmax/Softmax┬
IdentityIdentitysoftmax/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^p_re_lu/ReadVariableOp^p_re_lu_1/ReadVariableOp^p_re_lu_2/ReadVariableOp*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp20
p_re_lu/ReadVariableOpp_re_lu/ReadVariableOp24
p_re_lu_1/ReadVariableOpp_re_lu_1/ReadVariableOp24
p_re_lu_2/ReadVariableOpp_re_lu_2/ReadVariableOp:& "
 
_user_specified_nameinputs
▄
Е
(__inference_p_re_lu_layer_call_fn_865800

inputs"
statefulpartitionedcall_args_1
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         :Ю **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_p_re_lu_layer_call_and_return_conditional_losses_8657932
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         :Ю 2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4                                    :22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Я
є
+__inference_sequential_layer_call_fn_866082
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_8660682
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
ё7
╜
F__inference_sequential_layer_call_and_return_conditional_losses_866037
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2*
&p_re_lu_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2,
(p_re_lu_1_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2,
(p_re_lu_2_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвp_re_lu/StatefulPartitionedCallв!p_re_lu_1/StatefulPartitionedCallв!p_re_lu_2/StatefulPartitionedCallк
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         :Ю **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_8657722 
conv2d/StatefulPartitionedCallж
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0&p_re_lu_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         :Ю **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_p_re_lu_layer_call_and_return_conditional_losses_8657932!
p_re_lu/StatefulPartitionedCallў
max_pooling2d/PartitionedCallPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         O **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_8658062
max_pooling2d/PartitionedCall╥
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         K@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_8658242"
 conv2d_1/StatefulPartitionedCallп
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0(p_re_lu_1_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         K@**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_p_re_lu_1_layer_call_and_return_conditional_losses_8658452#
!p_re_lu_1/StatefulPartitionedCall 
max_pooling2d_1/PartitionedCallPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         %@**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8658582!
max_pooling2d_1/PartitionedCall╒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         !А**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_8658762"
 conv2d_2/StatefulPartitionedCall░
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0(p_re_lu_2_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         !А**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_p_re_lu_2_layer_call_and_return_conditional_losses_8658972#
!p_re_lu_2/StatefulPartitionedCallА
max_pooling2d_2/PartitionedCallPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8659102!
max_pooling2d_2/PartitionedCall▐
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8659442
flatten/PartitionedCall╡
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         (**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_8659622
dense/StatefulPartitionedCall▀
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_8659872
reshape/PartitionedCall┘
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_8660002
softmax/PartitionedCallщ
IdentityIdentity softmax/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
ш
┌
A__inference_dense_layer_call_and_return_conditional_losses_866338

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         (2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ш
┌
A__inference_dense_layer_call_and_return_conditional_losses_865962

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         (2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
є
ь
$__inference_signature_wrapper_866151
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identityИвStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__wrapped_model_8657602
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
ё

█
B__inference_conv2d_layer_call_and_return_conditional_losses_865772

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddп
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
▐
З
*__inference_p_re_lu_1_layer_call_fn_865852

inputs"
statefulpartitionedcall_args_1
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         K@**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_p_re_lu_1_layer_call_and_return_conditional_losses_8658452
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         K@2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4                                    :22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╟
J
.__inference_max_pooling2d_layer_call_fn_865812

inputs
identity╘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_8658062
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
└
к
)__inference_conv2d_1_layer_call_fn_865832

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_8658242
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Я
є
+__inference_sequential_layer_call_fn_866126
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_8661122
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
Аc
┘
!__inference__wrapped_model_865760
input_14
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource.
*sequential_p_re_lu_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource0
,sequential_p_re_lu_1_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource0
,sequential_p_re_lu_2_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identityИв(sequential/conv2d/BiasAdd/ReadVariableOpв'sequential/conv2d/Conv2D/ReadVariableOpв*sequential/conv2d_1/BiasAdd/ReadVariableOpв)sequential/conv2d_1/Conv2D/ReadVariableOpв*sequential/conv2d_2/BiasAdd/ReadVariableOpв)sequential/conv2d_2/Conv2D/ReadVariableOpв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв!sequential/p_re_lu/ReadVariableOpв#sequential/p_re_lu_1/ReadVariableOpв#sequential/p_re_lu_2/ReadVariableOp╦
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp▄
sequential/conv2d/Conv2DConv2Dinput_1/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         :Ю *
paddingVALID*
strides
2
sequential/conv2d/Conv2D┬
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp╤
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         :Ю 2
sequential/conv2d/BiasAddЩ
sequential/p_re_lu/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:         :Ю 2
sequential/p_re_lu/Relu╢
!sequential/p_re_lu/ReadVariableOpReadVariableOp*sequential_p_re_lu_readvariableop_resource*#
_output_shapes
::Ю *
dtype02#
!sequential/p_re_lu/ReadVariableOpР
sequential/p_re_lu/NegNeg)sequential/p_re_lu/ReadVariableOp:value:0*
T0*#
_output_shapes
::Ю 2
sequential/p_re_lu/NegЪ
sequential/p_re_lu/Neg_1Neg"sequential/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:         :Ю 2
sequential/p_re_lu/Neg_1Ч
sequential/p_re_lu/Relu_1Relusequential/p_re_lu/Neg_1:y:0*
T0*0
_output_shapes
:         :Ю 2
sequential/p_re_lu/Relu_1╖
sequential/p_re_lu/mulMulsequential/p_re_lu/Neg:y:0'sequential/p_re_lu/Relu_1:activations:0*
T0*0
_output_shapes
:         :Ю 2
sequential/p_re_lu/mul╖
sequential/p_re_lu/addAddV2%sequential/p_re_lu/Relu:activations:0sequential/p_re_lu/mul:z:0*
T0*0
_output_shapes
:         :Ю 2
sequential/p_re_lu/add╪
 sequential/max_pooling2d/MaxPoolMaxPoolsequential/p_re_lu/add:z:0*/
_output_shapes
:         O *
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool╤
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOpГ
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         K@*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D╚
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp╪
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         K@2
sequential/conv2d_1/BiasAddЮ
sequential/p_re_lu_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         K@2
sequential/p_re_lu_1/Relu╗
#sequential/p_re_lu_1/ReadVariableOpReadVariableOp,sequential_p_re_lu_1_readvariableop_resource*"
_output_shapes
:K@*
dtype02%
#sequential/p_re_lu_1/ReadVariableOpХ
sequential/p_re_lu_1/NegNeg+sequential/p_re_lu_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:K@2
sequential/p_re_lu_1/NegЯ
sequential/p_re_lu_1/Neg_1Neg$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         K@2
sequential/p_re_lu_1/Neg_1Ь
sequential/p_re_lu_1/Relu_1Relusequential/p_re_lu_1/Neg_1:y:0*
T0*/
_output_shapes
:         K@2
sequential/p_re_lu_1/Relu_1╛
sequential/p_re_lu_1/mulMulsequential/p_re_lu_1/Neg:y:0)sequential/p_re_lu_1/Relu_1:activations:0*
T0*/
_output_shapes
:         K@2
sequential/p_re_lu_1/mul╛
sequential/p_re_lu_1/addAddV2'sequential/p_re_lu_1/Relu:activations:0sequential/p_re_lu_1/mul:z:0*
T0*/
_output_shapes
:         K@2
sequential/p_re_lu_1/add▐
"sequential/max_pooling2d_1/MaxPoolMaxPoolsequential/p_re_lu_1/add:z:0*/
_output_shapes
:         %@*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool╥
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOpЖ
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         !А*
paddingVALID*
strides
2
sequential/conv2d_2/Conv2D╔
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp┘
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         !А2
sequential/conv2d_2/BiasAddЯ
sequential/p_re_lu_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         !А2
sequential/p_re_lu_2/Relu╝
#sequential/p_re_lu_2/ReadVariableOpReadVariableOp,sequential_p_re_lu_2_readvariableop_resource*#
_output_shapes
:!А*
dtype02%
#sequential/p_re_lu_2/ReadVariableOpЦ
sequential/p_re_lu_2/NegNeg+sequential/p_re_lu_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:!А2
sequential/p_re_lu_2/Negа
sequential/p_re_lu_2/Neg_1Neg$sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         !А2
sequential/p_re_lu_2/Neg_1Э
sequential/p_re_lu_2/Relu_1Relusequential/p_re_lu_2/Neg_1:y:0*
T0*0
_output_shapes
:         !А2
sequential/p_re_lu_2/Relu_1┐
sequential/p_re_lu_2/mulMulsequential/p_re_lu_2/Neg:y:0)sequential/p_re_lu_2/Relu_1:activations:0*
T0*0
_output_shapes
:         !А2
sequential/p_re_lu_2/mul┐
sequential/p_re_lu_2/addAddV2'sequential/p_re_lu_2/Relu:activations:0sequential/p_re_lu_2/mul:z:0*
T0*0
_output_shapes
:         !А2
sequential/p_re_lu_2/add▀
"sequential/max_pooling2d_2/MaxPoolMaxPoolsequential/p_re_lu_2/add:z:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPoolЕ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential/flatten/Const╞
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_2/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:         А@2
sequential/flatten/Reshape┴
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	А@(*
dtype02(
&sequential/dense/MatMul/ReadVariableOp├
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (2
sequential/dense/MatMul┐
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp┼
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         (2
sequential/dense/BiasAddЕ
sequential/reshape/ShapeShape!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential/reshape/ShapeЪ
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/reshape/strided_slice/stackЮ
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_1Ю
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_2╘
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/reshape/strided_sliceК
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/1К
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2$
"sequential/reshape/Reshape/shape/2 
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 sequential/reshape/Reshape/shape╟
sequential/reshape/ReshapeReshape!sequential/dense/BiasAdd:output:0)sequential/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         
2
sequential/reshape/ReshapeЮ
sequential/softmax/SoftmaxSoftmax#sequential/reshape/Reshape:output:0*
T0*+
_output_shapes
:         
2
sequential/softmax/Softmax╞
IdentityIdentity$sequential/softmax/Softmax:softmax:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp"^sequential/p_re_lu/ReadVariableOp$^sequential/p_re_lu_1/ReadVariableOp$^sequential/p_re_lu_2/ReadVariableOp*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2F
!sequential/p_re_lu/ReadVariableOp!sequential/p_re_lu/ReadVariableOp2J
#sequential/p_re_lu_1/ReadVariableOp#sequential/p_re_lu_1/ReadVariableOp2J
#sequential/p_re_lu_2/ReadVariableOp#sequential/p_re_lu_2/ReadVariableOp:' #
!
_user_specified_name	input_1
│
_
C__inference_reshape_layer_call_and_return_conditional_losses_865987

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         
2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*&
_input_shapes
:         (:& "
 
_user_specified_nameinputs
┘
D
(__inference_reshape_layer_call_fn_866363

inputs
identityп
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_8659872
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*&
_input_shapes
:         (:& "
 
_user_specified_nameinputs
Л
_
C__inference_flatten_layer_call_and_return_conditional_losses_866323

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
°

▌
D__inference_conv2d_2_layer_call_and_return_conditional_losses_865876

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOp╖
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd░
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
┬
к
)__inference_conv2d_2_layer_call_fn_865884

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_8658762
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ьP
з
__inference__traced_save_866517
file_prefix7
3savev2_sequential_conv2d_kernel_read_readvariableop5
1savev2_sequential_conv2d_bias_read_readvariableop7
3savev2_sequential_p_re_lu_alpha_read_readvariableop9
5savev2_sequential_conv2d_1_kernel_read_readvariableop7
3savev2_sequential_conv2d_1_bias_read_readvariableop9
5savev2_sequential_p_re_lu_1_alpha_read_readvariableop9
5savev2_sequential_conv2d_2_kernel_read_readvariableop7
3savev2_sequential_conv2d_2_bias_read_readvariableop9
5savev2_sequential_p_re_lu_2_alpha_read_readvariableop6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_adam_sequential_conv2d_kernel_m_read_readvariableop<
8savev2_adam_sequential_conv2d_bias_m_read_readvariableop>
:savev2_adam_sequential_p_re_lu_alpha_m_read_readvariableop@
<savev2_adam_sequential_conv2d_1_kernel_m_read_readvariableop>
:savev2_adam_sequential_conv2d_1_bias_m_read_readvariableop@
<savev2_adam_sequential_p_re_lu_1_alpha_m_read_readvariableop@
<savev2_adam_sequential_conv2d_2_kernel_m_read_readvariableop>
:savev2_adam_sequential_conv2d_2_bias_m_read_readvariableop@
<savev2_adam_sequential_p_re_lu_2_alpha_m_read_readvariableop=
9savev2_adam_sequential_dense_kernel_m_read_readvariableop;
7savev2_adam_sequential_dense_bias_m_read_readvariableop>
:savev2_adam_sequential_conv2d_kernel_v_read_readvariableop<
8savev2_adam_sequential_conv2d_bias_v_read_readvariableop>
:savev2_adam_sequential_p_re_lu_alpha_v_read_readvariableop@
<savev2_adam_sequential_conv2d_1_kernel_v_read_readvariableop>
:savev2_adam_sequential_conv2d_1_bias_v_read_readvariableop@
<savev2_adam_sequential_p_re_lu_1_alpha_v_read_readvariableop@
<savev2_adam_sequential_conv2d_2_kernel_v_read_readvariableop>
:savev2_adam_sequential_conv2d_2_bias_v_read_readvariableop@
<savev2_adam_sequential_p_re_lu_2_alpha_v_read_readvariableop=
9savev2_adam_sequential_dense_kernel_v_read_readvariableop;
7savev2_adam_sequential_dense_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1е
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_637a3bf7e52a474bb0441478d9827d8f/part2
StringJoin/inputs_1Б

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameм
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*╛
value┤B▒(B)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-1/alpha/.ATTRIBUTES/VARIABLE_VALUEB)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-4/alpha/.ATTRIBUTES/VARIABLE_VALUEB)layer-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-6/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-7/alpha/.ATTRIBUTES/VARIABLE_VALUEB*layer-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB(layer-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names╪
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╪
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_sequential_conv2d_kernel_read_readvariableop1savev2_sequential_conv2d_bias_read_readvariableop3savev2_sequential_p_re_lu_alpha_read_readvariableop5savev2_sequential_conv2d_1_kernel_read_readvariableop3savev2_sequential_conv2d_1_bias_read_readvariableop5savev2_sequential_p_re_lu_1_alpha_read_readvariableop5savev2_sequential_conv2d_2_kernel_read_readvariableop3savev2_sequential_conv2d_2_bias_read_readvariableop5savev2_sequential_p_re_lu_2_alpha_read_readvariableop2savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_adam_sequential_conv2d_kernel_m_read_readvariableop8savev2_adam_sequential_conv2d_bias_m_read_readvariableop:savev2_adam_sequential_p_re_lu_alpha_m_read_readvariableop<savev2_adam_sequential_conv2d_1_kernel_m_read_readvariableop:savev2_adam_sequential_conv2d_1_bias_m_read_readvariableop<savev2_adam_sequential_p_re_lu_1_alpha_m_read_readvariableop<savev2_adam_sequential_conv2d_2_kernel_m_read_readvariableop:savev2_adam_sequential_conv2d_2_bias_m_read_readvariableop<savev2_adam_sequential_p_re_lu_2_alpha_m_read_readvariableop9savev2_adam_sequential_dense_kernel_m_read_readvariableop7savev2_adam_sequential_dense_bias_m_read_readvariableop:savev2_adam_sequential_conv2d_kernel_v_read_readvariableop8savev2_adam_sequential_conv2d_bias_v_read_readvariableop:savev2_adam_sequential_p_re_lu_alpha_v_read_readvariableop<savev2_adam_sequential_conv2d_1_kernel_v_read_readvariableop:savev2_adam_sequential_conv2d_1_bias_v_read_readvariableop<savev2_adam_sequential_p_re_lu_1_alpha_v_read_readvariableop<savev2_adam_sequential_conv2d_2_kernel_v_read_readvariableop:savev2_adam_sequential_conv2d_2_bias_v_read_readvariableop<savev2_adam_sequential_p_re_lu_2_alpha_v_read_readvariableop9savev2_adam_sequential_dense_kernel_v_read_readvariableop7savev2_adam_sequential_dense_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*╝
_input_shapesк
з: : : ::Ю : @:@:K@:@А:А:!А:	А@(:(: : : : : : : : : ::Ю : @:@:K@:@А:А:!А:	А@(:(: : ::Ю : @:@:K@:@А:А:!А:	А@(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
р
З
*__inference_p_re_lu_2_layer_call_fn_865904

inputs"
statefulpartitionedcall_args_1
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         !А**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_p_re_lu_2_layer_call_and_return_conditional_losses_8658972
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         !А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4                                    :22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╥	
Т
E__inference_p_re_lu_1_layer_call_and_return_conditional_losses_865845

inputs
readvariableop_resource
identityИвReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4                                    2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:K@*
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:K@2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4                                    2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4                                    2
Relu_1j
mulMulNeg:y:0Relu_1:activations:0*
T0*/
_output_shapes
:         K@2
mulj
addAddV2Relu:activations:0mul:z:0*
T0*/
_output_shapes
:         K@2
addt
IdentityIdentityadd:z:0^ReadVariableOp*
T0*/
_output_shapes
:         K@2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4                                    :2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs
│
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_865806

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
╡
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_865858

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
Ь
Є
+__inference_sequential_layer_call_fn_866317

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identityИвStatefulPartitionedCall╖
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_8661122
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ь
Є
+__inference_sequential_layer_call_fn_866301

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identityИвStatefulPartitionedCall╖
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_8660682
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         
2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:         <а:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╕
serving_defaultд
D
input_19
serving_default_input_1:0         <а@
output_14
StatefulPartitionedCall:0         
tensorflow/serving/predict:ї┘
ЕE
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
▓_default_save_signature
+│&call_and_return_all_conditional_losses
┤__call__"ЛB
_tf_keras_sequentialьA{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [4, 10]}}, {"class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}}], "build_input_shape": [null, 60, 160, 1]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [4, 10]}}, {"class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}}], "build_input_shape": [null, 60, 160, 1]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ь

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+╡&call_and_return_all_conditional_losses
╢__call__"┼
_tf_keras_layerл{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
Э
	alpha
regularization_losses
trainable_variables
	variables
	keras_api
+╖&call_and_return_all_conditional_losses
╕__call__"Б
_tf_keras_layerч{"class_name": "PReLU", "name": "p_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
√
regularization_losses
 trainable_variables
!	variables
"	keras_api
+╣&call_and_return_all_conditional_losses
║__call__"ъ
_tf_keras_layer╨{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ё

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+╗&call_and_return_all_conditional_losses
╝__call__"╩
_tf_keras_layer░{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
б
	)alpha
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+╜&call_and_return_all_conditional_losses
╛__call__"Е
_tf_keras_layerы{"class_name": "PReLU", "name": "p_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 
.regularization_losses
/trainable_variables
0	variables
1	keras_api
+┐&call_and_return_all_conditional_losses
└__call__"ю
_tf_keras_layer╘{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Є

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+┴&call_and_return_all_conditional_losses
┬__call__"╦
_tf_keras_layer▒{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
б
	8alpha
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+├&call_and_return_all_conditional_losses
─__call__"Е
_tf_keras_layerы{"class_name": "PReLU", "name": "p_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 
=regularization_losses
>trainable_variables
?	variables
@	keras_api
+┼&call_and_return_all_conditional_losses
╞__call__"ю
_tf_keras_layer╘{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
о
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+╟&call_and_return_all_conditional_losses
╚__call__"Э
_tf_keras_layerГ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
є

Ekernel
Fbias
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
+╔&call_and_return_all_conditional_losses
╩__call__"╠
_tf_keras_layer▓{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}}}
Ч
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
+╦&call_and_return_all_conditional_losses
╠__call__"Ж
_tf_keras_layerь{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [4, 10]}}
К
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
+═&call_and_return_all_conditional_losses
╬__call__"∙
_tf_keras_layer▀{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}}
п
Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_ratemЬmЭmЮ#mЯ$mа)mб2mв3mг8mдEmеFmжvзvиvй#vк$vл)vм2vн3vо8vпEv░Fv▒"
	optimizer
 "
trackable_list_wrapper
n
0
1
2
#3
$4
)5
26
37
88
E9
F10"
trackable_list_wrapper
n
0
1
2
#3
$4
)5
26
37
88
E9
F10"
trackable_list_wrapper
╗
Xmetrics
regularization_losses
Ylayer_regularization_losses
Znon_trainable_variables
trainable_variables

[layers
	variables
┤__call__
▓_default_save_signature
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
-
╧serving_default"
signature_map
2:0 2sequential/conv2d/kernel
$:" 2sequential/conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Э
\metrics
regularization_losses
]layer_regularization_losses
trainable_variables
^non_trainable_variables

_layers
	variables
╢__call__
+╡&call_and_return_all_conditional_losses
'╡"call_and_return_conditional_losses"
_generic_user_object
/:-:Ю 2sequential/p_re_lu/alpha
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
Э
`metrics
regularization_losses
alayer_regularization_losses
trainable_variables
bnon_trainable_variables

clayers
	variables
╕__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
dmetrics
regularization_losses
elayer_regularization_losses
 trainable_variables
fnon_trainable_variables

glayers
!	variables
║__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
4:2 @2sequential/conv2d_1/kernel
&:$@2sequential/conv2d_1/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
Э
hmetrics
%regularization_losses
ilayer_regularization_losses
&trainable_variables
jnon_trainable_variables

klayers
'	variables
╝__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
0:.K@2sequential/p_re_lu_1/alpha
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
Э
lmetrics
*regularization_losses
mlayer_regularization_losses
+trainable_variables
nnon_trainable_variables

olayers
,	variables
╛__call__
+╜&call_and_return_all_conditional_losses
'╜"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
pmetrics
.regularization_losses
qlayer_regularization_losses
/trainable_variables
rnon_trainable_variables

slayers
0	variables
└__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
5:3@А2sequential/conv2d_2/kernel
':%А2sequential/conv2d_2/bias
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
Э
tmetrics
4regularization_losses
ulayer_regularization_losses
5trainable_variables
vnon_trainable_variables

wlayers
6	variables
┬__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
1:/!А2sequential/p_re_lu_2/alpha
 "
trackable_list_wrapper
'
80"
trackable_list_wrapper
'
80"
trackable_list_wrapper
Э
xmetrics
9regularization_losses
ylayer_regularization_losses
:trainable_variables
znon_trainable_variables

{layers
;	variables
─__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
|metrics
=regularization_losses
}layer_regularization_losses
>trainable_variables
~non_trainable_variables

layers
?	variables
╞__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
Аmetrics
Aregularization_losses
 Бlayer_regularization_losses
Btrainable_variables
Вnon_trainable_variables
Гlayers
C	variables
╚__call__
+╟&call_and_return_all_conditional_losses
'╟"call_and_return_conditional_losses"
_generic_user_object
*:(	А@(2sequential/dense/kernel
#:!(2sequential/dense/bias
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
б
Дmetrics
Gregularization_losses
 Еlayer_regularization_losses
Htrainable_variables
Жnon_trainable_variables
Зlayers
I	variables
╩__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
Иmetrics
Kregularization_losses
 Йlayer_regularization_losses
Ltrainable_variables
Кnon_trainable_variables
Лlayers
M	variables
╠__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
Мmetrics
Oregularization_losses
 Нlayer_regularization_losses
Ptrainable_variables
Оnon_trainable_variables
Пlayers
Q	variables
╬__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
Р0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
г

Сtotal

Тcount
У
_fn_kwargs
Фregularization_losses
Хtrainable_variables
Ц	variables
Ч	keras_api
+╨&call_and_return_all_conditional_losses
╤__call__"х
_tf_keras_layer╦{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
д
Шmetrics
Фregularization_losses
 Щlayer_regularization_losses
Хtrainable_variables
Ъnon_trainable_variables
Ыlayers
Ц	variables
╤__call__
+╨&call_and_return_all_conditional_losses
'╨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
 "
trackable_list_wrapper
7:5 2Adam/sequential/conv2d/kernel/m
):' 2Adam/sequential/conv2d/bias/m
4:2:Ю 2Adam/sequential/p_re_lu/alpha/m
9:7 @2!Adam/sequential/conv2d_1/kernel/m
+:)@2Adam/sequential/conv2d_1/bias/m
5:3K@2!Adam/sequential/p_re_lu_1/alpha/m
::8@А2!Adam/sequential/conv2d_2/kernel/m
,:*А2Adam/sequential/conv2d_2/bias/m
6:4!А2!Adam/sequential/p_re_lu_2/alpha/m
/:-	А@(2Adam/sequential/dense/kernel/m
(:&(2Adam/sequential/dense/bias/m
7:5 2Adam/sequential/conv2d/kernel/v
):' 2Adam/sequential/conv2d/bias/v
4:2:Ю 2Adam/sequential/p_re_lu/alpha/v
9:7 @2!Adam/sequential/conv2d_1/kernel/v
+:)@2Adam/sequential/conv2d_1/bias/v
5:3K@2!Adam/sequential/p_re_lu_1/alpha/v
::8@А2!Adam/sequential/conv2d_2/kernel/v
,:*А2Adam/sequential/conv2d_2/bias/v
6:4!А2!Adam/sequential/p_re_lu_2/alpha/v
/:-	А@(2Adam/sequential/dense/kernel/v
(:&(2Adam/sequential/dense/bias/v
ш2х
!__inference__wrapped_model_865760┐
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк */в,
*К'
input_1         <а
ц2у
F__inference_sequential_layer_call_and_return_conditional_losses_866218
F__inference_sequential_layer_call_and_return_conditional_losses_866285
F__inference_sequential_layer_call_and_return_conditional_losses_866009
F__inference_sequential_layer_call_and_return_conditional_losses_866037└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
·2ў
+__inference_sequential_layer_call_fn_866301
+__inference_sequential_layer_call_fn_866317
+__inference_sequential_layer_call_fn_866082
+__inference_sequential_layer_call_fn_866126└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
б2Ю
B__inference_conv2d_layer_call_and_return_conditional_losses_865772╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Ж2Г
'__inference_conv2d_layer_call_fn_865780╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
л2и
C__inference_p_re_lu_layer_call_and_return_conditional_losses_865793р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Р2Н
(__inference_p_re_lu_layer_call_fn_865800р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▒2о
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_865806р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ц2У
.__inference_max_pooling2d_layer_call_fn_865812р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
г2а
D__inference_conv2d_1_layer_call_and_return_conditional_losses_865824╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
И2Е
)__inference_conv2d_1_layer_call_fn_865832╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
н2к
E__inference_p_re_lu_1_layer_call_and_return_conditional_losses_865845р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Т2П
*__inference_p_re_lu_1_layer_call_fn_865852р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
│2░
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_865858р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_1_layer_call_fn_865864р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
г2а
D__inference_conv2d_2_layer_call_and_return_conditional_losses_865876╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
И2Е
)__inference_conv2d_2_layer_call_fn_865884╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
н2к
E__inference_p_re_lu_2_layer_call_and_return_conditional_losses_865897р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Т2П
*__inference_p_re_lu_2_layer_call_fn_865904р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
│2░
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_865910р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_2_layer_call_fn_865916р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
э2ъ
C__inference_flatten_layer_call_and_return_conditional_losses_866323в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_flatten_layer_call_fn_866328в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_layer_call_and_return_conditional_losses_866338в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_layer_call_fn_866345в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_reshape_layer_call_and_return_conditional_losses_866358в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_reshape_layer_call_fn_866363в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_softmax_layer_call_and_return_conditional_losses_866368в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_softmax_layer_call_fn_866373в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
3B1
$__inference_signature_wrapper_866151input_1
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 з
!__inference__wrapped_model_865760Б#$)238EF9в6
/в,
*К'
input_1         <а
к "7к4
2
output_1&К#
output_1         
┘
D__inference_conv2d_1_layer_call_and_return_conditional_losses_865824Р#$IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                           @
Ъ ▒
)__inference_conv2d_1_layer_call_fn_865832Г#$IвF
?в<
:К7
inputs+                            
к "2К/+                           @┌
D__inference_conv2d_2_layer_call_and_return_conditional_losses_865876С23IвF
?в<
:К7
inputs+                           @
к "@в=
6К3
0,                           А
Ъ ▓
)__inference_conv2d_2_layer_call_fn_865884Д23IвF
?в<
:К7
inputs+                           @
к "3К0,                           А╫
B__inference_conv2d_layer_call_and_return_conditional_losses_865772РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                            
Ъ п
'__inference_conv2d_layer_call_fn_865780ГIвF
?в<
:К7
inputs+                           
к "2К/+                            в
A__inference_dense_layer_call_and_return_conditional_losses_866338]EF0в-
&в#
!К
inputs         А@
к "%в"
К
0         (
Ъ z
&__inference_dense_layer_call_fn_866345PEF0в-
&в#
!К
inputs         А@
к "К         (й
C__inference_flatten_layer_call_and_return_conditional_losses_866323b8в5
.в+
)К&
inputs         А
к "&в#
К
0         А@
Ъ Б
(__inference_flatten_layer_call_fn_866328U8в5
.в+
)К&
inputs         А
к "К         А@ю
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_865858ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_1_layer_call_fn_865864СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_865910ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_2_layer_call_fn_865916СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ь
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_865806ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ─
.__inference_max_pooling2d_layer_call_fn_865812СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╨
E__inference_p_re_lu_1_layer_call_and_return_conditional_losses_865845Ж)RвO
HвE
CК@
inputs4                                    
к "-в*
#К 
0         K@
Ъ з
*__inference_p_re_lu_1_layer_call_fn_865852y)RвO
HвE
CК@
inputs4                                    
к " К         K@╤
E__inference_p_re_lu_2_layer_call_and_return_conditional_losses_865897З8RвO
HвE
CК@
inputs4                                    
к ".в+
$К!
0         !А
Ъ и
*__inference_p_re_lu_2_layer_call_fn_865904z8RвO
HвE
CК@
inputs4                                    
к "!К         !А╧
C__inference_p_re_lu_layer_call_and_return_conditional_losses_865793ЗRвO
HвE
CК@
inputs4                                    
к ".в+
$К!
0         :Ю 
Ъ ж
(__inference_p_re_lu_layer_call_fn_865800zRвO
HвE
CК@
inputs4                                    
к "!К         :Ю г
C__inference_reshape_layer_call_and_return_conditional_losses_866358\/в,
%в"
 К
inputs         (
к ")в&
К
0         

Ъ {
(__inference_reshape_layer_call_fn_866363O/в,
%в"
 К
inputs         (
к "К         
┼
F__inference_sequential_layer_call_and_return_conditional_losses_866009{#$)238EFAв>
7в4
*К'
input_1         <а
p

 
к ")в&
К
0         

Ъ ┼
F__inference_sequential_layer_call_and_return_conditional_losses_866037{#$)238EFAв>
7в4
*К'
input_1         <а
p 

 
к ")в&
К
0         

Ъ ─
F__inference_sequential_layer_call_and_return_conditional_losses_866218z#$)238EF@в=
6в3
)К&
inputs         <а
p

 
к ")в&
К
0         

Ъ ─
F__inference_sequential_layer_call_and_return_conditional_losses_866285z#$)238EF@в=
6в3
)К&
inputs         <а
p 

 
к ")в&
К
0         

Ъ Э
+__inference_sequential_layer_call_fn_866082n#$)238EFAв>
7в4
*К'
input_1         <а
p

 
к "К         
Э
+__inference_sequential_layer_call_fn_866126n#$)238EFAв>
7в4
*К'
input_1         <а
p 

 
к "К         
Ь
+__inference_sequential_layer_call_fn_866301m#$)238EF@в=
6в3
)К&
inputs         <а
p

 
к "К         
Ь
+__inference_sequential_layer_call_fn_866317m#$)238EF@в=
6в3
)К&
inputs         <а
p 

 
к "К         
╡
$__inference_signature_wrapper_866151М#$)238EFDвA
в 
:к7
5
input_1*К'
input_1         <а"7к4
2
output_1&К#
output_1         
з
C__inference_softmax_layer_call_and_return_conditional_losses_866368`3в0
)в&
$К!
inputs         

к ")в&
К
0         

Ъ 
(__inference_softmax_layer_call_fn_866373S3в0
)в&
$К!
inputs         

к "К         
