Łž
ÍŁ
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
dtypetype
ž
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ţć

custom_actor/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ĺ**
shared_namecustom_actor/dense/kernel

-custom_actor/dense/kernel/Read/ReadVariableOpReadVariableOpcustom_actor/dense/kernel* 
_output_shapes
:
Ĺ*
dtype0

custom_actor/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namecustom_actor/dense/bias

+custom_actor/dense/bias/Read/ReadVariableOpReadVariableOpcustom_actor/dense/bias*
_output_shapes	
:*
dtype0

custom_actor/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namecustom_actor/dense_1/kernel

/custom_actor/dense_1/kernel/Read/ReadVariableOpReadVariableOpcustom_actor/dense_1/kernel* 
_output_shapes
:
*
dtype0

custom_actor/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecustom_actor/dense_1/bias

-custom_actor/dense_1/bias/Read/ReadVariableOpReadVariableOpcustom_actor/dense_1/bias*
_output_shapes	
:*
dtype0

custom_actor/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	$*,
shared_namecustom_actor/dense_2/kernel

/custom_actor/dense_2/kernel/Read/ReadVariableOpReadVariableOpcustom_actor/dense_2/kernel*
_output_shapes
:	$*
dtype0

custom_actor/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_namecustom_actor/dense_2/bias

-custom_actor/dense_2/bias/Read/ReadVariableOpReadVariableOpcustom_actor/dense_2/bias*
_output_shapes
:$*
dtype0

NoOpNoOp
É
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueúB÷ Bđ
y
d1
d2
a
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*
	0

1
2
3
4
5
 
*
	0

1
2
3
4
5
­
non_trainable_variables
metrics
	variables
layer_metrics
regularization_losses
trainable_variables
layer_regularization_losses

layers
 
SQ
VARIABLE_VALUEcustom_actor/dense/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcustom_actor/dense/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1
 

	0

1
­
 non_trainable_variables
!metrics
	variables
"layer_metrics
regularization_losses
trainable_variables
#layer_regularization_losses

$layers
US
VARIABLE_VALUEcustom_actor/dense_1/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcustom_actor/dense_1/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
%non_trainable_variables
&metrics
	variables
'layer_metrics
regularization_losses
trainable_variables
(layer_regularization_losses

)layers
TR
VARIABLE_VALUEcustom_actor/dense_2/kernel#a/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcustom_actor/dense_2/bias!a/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
*non_trainable_variables
+metrics
	variables
,layer_metrics
regularization_losses
trainable_variables
-layer_regularization_losses

.layers
 
 
 
 

0
1
2
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
|
serving_default_input_1Placeholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ*
dtype0*
shape:˙˙˙˙˙˙˙˙˙Ĺ
ĺ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1custom_actor/dense/kernelcustom_actor/dense/biascustom_actor/dense_1/kernelcustom_actor/dense_1/biascustom_actor/dense_2/kernelcustom_actor/dense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙$*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_152349544
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ŕ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-custom_actor/dense/kernel/Read/ReadVariableOp+custom_actor/dense/bias/Read/ReadVariableOp/custom_actor/dense_1/kernel/Read/ReadVariableOp-custom_actor/dense_1/bias/Read/ReadVariableOp/custom_actor/dense_2/kernel/Read/ReadVariableOp-custom_actor/dense_2/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_save_152349644
Ă
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecustom_actor/dense/kernelcustom_actor/dense/biascustom_actor/dense_1/kernelcustom_actor/dense_1/biascustom_actor/dense_2/kernelcustom_actor/dense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__traced_restore_152349672šź
˛
Ź
D__inference_dense_layer_call_and_return_conditional_losses_152349555

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ĺ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙Ĺ:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
 
_user_specified_nameinputs
đ
Â
0__inference_custom_actor_layer_call_fn_152349525
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙$*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_custom_actor_layer_call_and_return_conditional_losses_1523495072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:˙˙˙˙˙˙˙˙˙Ĺ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
!
_user_specified_name	input_1
ŕ
~
)__inference_dense_layer_call_fn_152349564

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallő
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_1523494372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙Ĺ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
 
_user_specified_nameinputs
´
Ž
F__inference_dense_1_layer_call_and_return_conditional_losses_152349575

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
´
Ž
F__inference_dense_1_layer_call_and_return_conditional_losses_152349464

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ň
Ž
F__inference_dense_2_layer_call_and_return_conditional_losses_152349594

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ň
Ž
F__inference_dense_2_layer_call_and_return_conditional_losses_152349490

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˛
Ź
D__inference_dense_layer_call_and_return_conditional_losses_152349437

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ĺ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙Ĺ:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
 
_user_specified_nameinputs
ă

+__inference_dense_2_layer_call_fn_152349603

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_1523494902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ş

$__inference__wrapped_model_152349422
input_15
1custom_actor_dense_matmul_readvariableop_resource6
2custom_actor_dense_biasadd_readvariableop_resource7
3custom_actor_dense_1_matmul_readvariableop_resource8
4custom_actor_dense_1_biasadd_readvariableop_resource7
3custom_actor_dense_2_matmul_readvariableop_resource8
4custom_actor_dense_2_biasadd_readvariableop_resource
identityČ
(custom_actor/dense/MatMul/ReadVariableOpReadVariableOp1custom_actor_dense_matmul_readvariableop_resource* 
_output_shapes
:
Ĺ*
dtype02*
(custom_actor/dense/MatMul/ReadVariableOpŽ
custom_actor/dense/MatMulMatMulinput_10custom_actor/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_actor/dense/MatMulĆ
)custom_actor/dense/BiasAdd/ReadVariableOpReadVariableOp2custom_actor_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)custom_actor/dense/BiasAdd/ReadVariableOpÎ
custom_actor/dense/BiasAddBiasAdd#custom_actor/dense/MatMul:product:01custom_actor/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_actor/dense/BiasAdd
custom_actor/dense/ReluRelu#custom_actor/dense/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_actor/dense/ReluÎ
*custom_actor/dense_1/MatMul/ReadVariableOpReadVariableOp3custom_actor_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*custom_actor/dense_1/MatMul/ReadVariableOpŇ
custom_actor/dense_1/MatMulMatMul%custom_actor/dense/Relu:activations:02custom_actor/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_actor/dense_1/MatMulĚ
+custom_actor/dense_1/BiasAdd/ReadVariableOpReadVariableOp4custom_actor_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+custom_actor/dense_1/BiasAdd/ReadVariableOpÖ
custom_actor/dense_1/BiasAddBiasAdd%custom_actor/dense_1/MatMul:product:03custom_actor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_actor/dense_1/BiasAdd
custom_actor/dense_1/ReluRelu%custom_actor/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_actor/dense_1/ReluÍ
*custom_actor/dense_2/MatMul/ReadVariableOpReadVariableOp3custom_actor_dense_2_matmul_readvariableop_resource*
_output_shapes
:	$*
dtype02,
*custom_actor/dense_2/MatMul/ReadVariableOpÓ
custom_actor/dense_2/MatMulMatMul'custom_actor/dense_1/Relu:activations:02custom_actor/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2
custom_actor/dense_2/MatMulË
+custom_actor/dense_2/BiasAdd/ReadVariableOpReadVariableOp4custom_actor_dense_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02-
+custom_actor/dense_2/BiasAdd/ReadVariableOpŐ
custom_actor/dense_2/BiasAddBiasAdd%custom_actor/dense_2/MatMul:product:03custom_actor/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2
custom_actor/dense_2/BiasAddy
IdentityIdentity%custom_actor/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:˙˙˙˙˙˙˙˙˙Ĺ:::::::Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
!
_user_specified_name	input_1
Ś
Í
"__inference__traced_save_152349644
file_prefix8
4savev2_custom_actor_dense_kernel_read_readvariableop6
2savev2_custom_actor_dense_bias_read_readvariableop:
6savev2_custom_actor_dense_1_kernel_read_readvariableop8
4savev2_custom_actor_dense_1_bias_read_readvariableop:
6savev2_custom_actor_dense_2_kernel_read_readvariableop8
4savev2_custom_actor_dense_2_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e328339e524641bbab509cbbde88c3a5/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB#a/kernel/.ATTRIBUTES/VARIABLE_VALUEB!a/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_custom_actor_dense_kernel_read_readvariableop2savev2_custom_actor_dense_bias_read_readvariableop6savev2_custom_actor_dense_1_kernel_read_readvariableop4savev2_custom_actor_dense_1_bias_read_readvariableop6savev2_custom_actor_dense_2_kernel_read_readvariableop4savev2_custom_actor_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*N
_input_shapes=
;: :
Ĺ::
::	$:$: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
Ĺ:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	$: 

_output_shapes
:$:

_output_shapes
: 
ĺ

+__inference_dense_1_layer_call_fn_152349584

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_1523494642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Â
Ő
K__inference_custom_actor_layer_call_and_return_conditional_losses_152349507
input_1
dense_152349448
dense_152349450
dense_1_152349475
dense_1_152349477
dense_2_152349501
dense_2_152349503
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_152349448dense_152349450*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_1523494372
dense/StatefulPartitionedCallš
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_152349475dense_1_152349477*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_1523494642!
dense_1/StatefulPartitionedCallş
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_152349501dense_2_152349503*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_1523494902!
dense_2/StatefulPartitionedCallŕ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:˙˙˙˙˙˙˙˙˙Ĺ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
!
_user_specified_name	input_1
˛
ó
%__inference__traced_restore_152349672
file_prefix.
*assignvariableop_custom_actor_dense_kernel.
*assignvariableop_1_custom_actor_dense_bias2
.assignvariableop_2_custom_actor_dense_1_kernel0
,assignvariableop_3_custom_actor_dense_1_bias2
.assignvariableop_4_custom_actor_dense_2_kernel0
,assignvariableop_5_custom_actor_dense_2_bias

identity_7˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB#a/kernel/.ATTRIBUTES/VARIABLE_VALUEB!a/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityŠ
AssignVariableOpAssignVariableOp*assignvariableop_custom_actor_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ż
AssignVariableOp_1AssignVariableOp*assignvariableop_1_custom_actor_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ł
AssignVariableOp_2AssignVariableOp.assignvariableop_2_custom_actor_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ą
AssignVariableOp_3AssignVariableOp,assignvariableop_3_custom_actor_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ł
AssignVariableOp_4AssignVariableOp.assignvariableop_4_custom_actor_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ą
AssignVariableOp_5AssignVariableOp,assignvariableop_5_custom_actor_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpä

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6Ö

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ŕ
š
'__inference_signature_wrapper_152349544
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙$*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__wrapped_model_1523494222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙$2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:˙˙˙˙˙˙˙˙˙Ĺ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
!
_user_specified_name	input_1"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ź
serving_default
<
input_11
serving_default_input_1:0˙˙˙˙˙˙˙˙˙Ĺ<
output_10
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙$tensorflow/serving/predict:đO
Ů
d1
d2
a
	variables
regularization_losses
trainable_variables
	keras_api

signatures
*/&call_and_return_all_conditional_losses
0_default_save_signature
1__call__"
_tf_keras_modelě{"class_name": "custom_actor", "name": "custom_actor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "custom_actor"}}
í

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
*2&call_and_return_all_conditional_losses
3__call__"Č
_tf_keras_layerŽ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 197}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 197]}}
ň

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*4&call_and_return_all_conditional_losses
5__call__"Í
_tf_keras_layerł{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1024]}}
ń

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*6&call_and_return_all_conditional_losses
7__call__"Ě
_tf_keras_layer˛{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
J
	0

1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
Ę
non_trainable_variables
metrics
	variables
layer_metrics
regularization_losses
trainable_variables
layer_regularization_losses

layers
1__call__
0_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
,
8serving_default"
signature_map
-:+
Ĺ2custom_actor/dense/kernel
&:$2custom_actor/dense/bias
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
­
 non_trainable_variables
!metrics
	variables
"layer_metrics
regularization_losses
trainable_variables
#layer_regularization_losses

$layers
3__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
/:-
2custom_actor/dense_1/kernel
(:&2custom_actor/dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
%non_trainable_variables
&metrics
	variables
'layer_metrics
regularization_losses
trainable_variables
(layer_regularization_losses

)layers
5__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
.:,	$2custom_actor/dense_2/kernel
':%$2custom_actor/dense_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
*non_trainable_variables
+metrics
	variables
,layer_metrics
regularization_losses
trainable_variables
-layer_regularization_losses

.layers
7__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
K__inference_custom_actor_layer_call_and_return_conditional_losses_152349507Ë
˛
FullArgSpec!
args
jself
j
input_data
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *'˘$
"
input_1˙˙˙˙˙˙˙˙˙Ĺ
ă2ŕ
$__inference__wrapped_model_152349422ˇ
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *'˘$
"
input_1˙˙˙˙˙˙˙˙˙Ĺ
2
0__inference_custom_actor_layer_call_fn_152349525Ë
˛
FullArgSpec!
args
jself
j
input_data
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *'˘$
"
input_1˙˙˙˙˙˙˙˙˙Ĺ
î2ë
D__inference_dense_layer_call_and_return_conditional_losses_152349555˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ó2Đ
)__inference_dense_layer_call_fn_152349564˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
F__inference_dense_1_layer_call_and_return_conditional_losses_152349575˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_dense_1_layer_call_fn_152349584˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
F__inference_dense_2_layer_call_and_return_conditional_losses_152349594˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_dense_2_layer_call_fn_152349603˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
6B4
'__inference_signature_wrapper_152349544input_1
$__inference__wrapped_model_152349422p	
1˘.
'˘$
"
input_1˙˙˙˙˙˙˙˙˙Ĺ
Ş "3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙$ą
K__inference_custom_actor_layer_call_and_return_conditional_losses_152349507b	
1˘.
'˘$
"
input_1˙˙˙˙˙˙˙˙˙Ĺ
Ş "%˘"

0˙˙˙˙˙˙˙˙˙$
 
0__inference_custom_actor_layer_call_fn_152349525U	
1˘.
'˘$
"
input_1˙˙˙˙˙˙˙˙˙Ĺ
Ş "˙˙˙˙˙˙˙˙˙$¨
F__inference_dense_1_layer_call_and_return_conditional_losses_152349575^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
+__inference_dense_1_layer_call_fn_152349584Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙§
F__inference_dense_2_layer_call_and_return_conditional_losses_152349594]0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙$
 
+__inference_dense_2_layer_call_fn_152349603P0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙$Ś
D__inference_dense_layer_call_and_return_conditional_losses_152349555^	
0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙Ĺ
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
)__inference_dense_layer_call_fn_152349564Q	
0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙Ĺ
Ş "˙˙˙˙˙˙˙˙˙Ś
'__inference_signature_wrapper_152349544{	
<˘9
˘ 
2Ş/
-
input_1"
input_1˙˙˙˙˙˙˙˙˙Ĺ"3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙$