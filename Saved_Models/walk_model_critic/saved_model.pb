ż
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
 "serve*2.3.12v2.3.0-54-gfcc4b966f18č

custom_critic/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ĺ*-
shared_namecustom_critic/dense_3/kernel

0custom_critic/dense_3/kernel/Read/ReadVariableOpReadVariableOpcustom_critic/dense_3/kernel* 
_output_shapes
:
Ĺ*
dtype0

custom_critic/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namecustom_critic/dense_3/bias

.custom_critic/dense_3/bias/Read/ReadVariableOpReadVariableOpcustom_critic/dense_3/bias*
_output_shapes	
:*
dtype0

custom_critic/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namecustom_critic/dense_4/kernel

0custom_critic/dense_4/kernel/Read/ReadVariableOpReadVariableOpcustom_critic/dense_4/kernel* 
_output_shapes
:
*
dtype0

custom_critic/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namecustom_critic/dense_4/bias

.custom_critic/dense_4/bias/Read/ReadVariableOpReadVariableOpcustom_critic/dense_4/bias*
_output_shapes	
:*
dtype0

custom_critic/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namecustom_critic/dense_5/kernel

0custom_critic/dense_5/kernel/Read/ReadVariableOpReadVariableOpcustom_critic/dense_5/kernel*
_output_shapes
:	*
dtype0

custom_critic/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namecustom_critic/dense_5/bias

.custom_critic/dense_5/bias/Read/ReadVariableOpReadVariableOpcustom_critic/dense_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ó
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bú
y
d1
d2
v
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

	kernel

bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
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
*
	0

1
2
3
4
5
 
­
metrics
layer_metrics
trainable_variables

layers
	variables
regularization_losses
non_trainable_variables
layer_regularization_losses
 
VT
VARIABLE_VALUEcustom_critic/dense_3/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcustom_critic/dense_3/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
­
 metrics
regularization_losses
!layer_metrics

"layers
	variables
trainable_variables
#non_trainable_variables
$layer_regularization_losses
VT
VARIABLE_VALUEcustom_critic/dense_4/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcustom_critic/dense_4/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
%metrics
regularization_losses
&layer_metrics

'layers
	variables
trainable_variables
(non_trainable_variables
)layer_regularization_losses
US
VARIABLE_VALUEcustom_critic/dense_5/kernel#v/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcustom_critic/dense_5/bias!v/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
*metrics
regularization_losses
+layer_metrics

,layers
	variables
trainable_variables
-non_trainable_variables
.layer_regularization_losses
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
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ*
dtype0*
shape:˙˙˙˙˙˙˙˙˙Ĺ
ě
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1custom_critic/dense_3/kernelcustom_critic/dense_3/biascustom_critic/dense_4/kernelcustom_critic/dense_4/biascustom_critic/dense_5/kernelcustom_critic/dense_5/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_695529
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ç
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0custom_critic/dense_3/kernel/Read/ReadVariableOp.custom_critic/dense_3/bias/Read/ReadVariableOp0custom_critic/dense_4/kernel/Read/ReadVariableOp.custom_critic/dense_4/bias/Read/ReadVariableOp0custom_critic/dense_5/kernel/Read/ReadVariableOp.custom_critic/dense_5/bias/Read/ReadVariableOpConst*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_695629
Ę
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecustom_critic/dense_3/kernelcustom_critic/dense_3/biascustom_critic/dense_4/kernelcustom_critic/dense_4/biascustom_critic/dense_5/kernelcustom_critic/dense_5/bias*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_695657őź
ą
Ť
C__inference_dense_3_layer_call_and_return_conditional_losses_695540

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
Ă
ú
"__inference__traced_restore_695657
file_prefix1
-assignvariableop_custom_critic_dense_3_kernel1
-assignvariableop_1_custom_critic_dense_3_bias3
/assignvariableop_2_custom_critic_dense_4_kernel1
-assignvariableop_3_custom_critic_dense_4_bias3
/assignvariableop_4_custom_critic_dense_5_kernel1
-assignvariableop_5_custom_critic_dense_5_bias

identity_7˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB#v/kernel/.ATTRIBUTES/VARIABLE_VALUEB!v/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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

IdentityŹ
AssignVariableOpAssignVariableOp-assignvariableop_custom_critic_dense_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1˛
AssignVariableOp_1AssignVariableOp-assignvariableop_1_custom_critic_dense_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2´
AssignVariableOp_2AssignVariableOp/assignvariableop_2_custom_critic_dense_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3˛
AssignVariableOp_3AssignVariableOp-assignvariableop_3_custom_critic_dense_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4´
AssignVariableOp_4AssignVariableOp/assignvariableop_4_custom_critic_dense_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5˛
AssignVariableOp_5AssignVariableOp-assignvariableop_5_custom_critic_dense_5_biasIdentity_5:output:0"/device:CPU:0*
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

 
!__inference__wrapped_model_695407
input_18
4custom_critic_dense_3_matmul_readvariableop_resource9
5custom_critic_dense_3_biasadd_readvariableop_resource8
4custom_critic_dense_4_matmul_readvariableop_resource9
5custom_critic_dense_4_biasadd_readvariableop_resource8
4custom_critic_dense_5_matmul_readvariableop_resource9
5custom_critic_dense_5_biasadd_readvariableop_resource
identityŃ
+custom_critic/dense_3/MatMul/ReadVariableOpReadVariableOp4custom_critic_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
Ĺ*
dtype02-
+custom_critic/dense_3/MatMul/ReadVariableOpˇ
custom_critic/dense_3/MatMulMatMulinput_13custom_critic/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_critic/dense_3/MatMulĎ
,custom_critic/dense_3/BiasAdd/ReadVariableOpReadVariableOp5custom_critic_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,custom_critic/dense_3/BiasAdd/ReadVariableOpÚ
custom_critic/dense_3/BiasAddBiasAdd&custom_critic/dense_3/MatMul:product:04custom_critic/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_critic/dense_3/BiasAdd
custom_critic/dense_3/ReluRelu&custom_critic/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_critic/dense_3/ReluŃ
+custom_critic/dense_4/MatMul/ReadVariableOpReadVariableOp4custom_critic_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02-
+custom_critic/dense_4/MatMul/ReadVariableOpŘ
custom_critic/dense_4/MatMulMatMul(custom_critic/dense_3/Relu:activations:03custom_critic/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_critic/dense_4/MatMulĎ
,custom_critic/dense_4/BiasAdd/ReadVariableOpReadVariableOp5custom_critic_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,custom_critic/dense_4/BiasAdd/ReadVariableOpÚ
custom_critic/dense_4/BiasAddBiasAdd&custom_critic/dense_4/MatMul:product:04custom_critic/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_critic/dense_4/BiasAdd
custom_critic/dense_4/ReluRelu&custom_critic/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_critic/dense_4/ReluĐ
+custom_critic/dense_5/MatMul/ReadVariableOpReadVariableOp4custom_critic_dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+custom_critic/dense_5/MatMul/ReadVariableOp×
custom_critic/dense_5/MatMulMatMul(custom_critic/dense_4/Relu:activations:03custom_critic/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_critic/dense_5/MatMulÎ
,custom_critic/dense_5/BiasAdd/ReadVariableOpReadVariableOp5custom_critic_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,custom_critic/dense_5/BiasAdd/ReadVariableOpŮ
custom_critic/dense_5/BiasAddBiasAdd&custom_critic/dense_5/MatMul:product:04custom_critic/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
custom_critic/dense_5/BiasAddz
IdentityIdentity&custom_critic/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:˙˙˙˙˙˙˙˙˙Ĺ:::::::Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
!
_user_specified_name	input_1
Ť
Ç
I__inference_custom_critic_layer_call_and_return_conditional_losses_695492
input_1
dense_3_695433
dense_3_695435
dense_4_695460
dense_4_695462
dense_5_695486
dense_5_695488
identity˘dense_3/StatefulPartitionedCall˘dense_4/StatefulPartitionedCall˘dense_5/StatefulPartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3_695433dense_3_695435*
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
GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6954222!
dense_3/StatefulPartitionedCall˛
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_695460dense_4_695462*
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
GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6954492!
dense_4/StatefulPartitionedCallą
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_695486dense_5_695488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_6954752!
dense_5/StatefulPartitionedCallâ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:˙˙˙˙˙˙˙˙˙Ĺ::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Q M
(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
!
_user_specified_name	input_1
ą
Ť
C__inference_dense_4_layer_call_and_return_conditional_losses_695449

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
Ď
Ť
C__inference_dense_5_layer_call_and_return_conditional_losses_695475

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ˇ
Ô
__inference__traced_save_695629
file_prefix;
7savev2_custom_critic_dense_3_kernel_read_readvariableop9
5savev2_custom_critic_dense_3_bias_read_readvariableop;
7savev2_custom_critic_dense_4_kernel_read_readvariableop9
5savev2_custom_critic_dense_4_bias_read_readvariableop;
7savev2_custom_critic_dense_5_kernel_read_readvariableop9
5savev2_custom_critic_dense_5_bias_read_readvariableop
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
value3B1 B+_temp_6c9de1640043421997e24abb70664940/part2	
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
valueBB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB#v/kernel/.ATTRIBUTES/VARIABLE_VALUEB!v/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_custom_critic_dense_3_kernel_read_readvariableop5savev2_custom_critic_dense_3_bias_read_readvariableop7savev2_custom_critic_dense_4_kernel_read_readvariableop5savev2_custom_critic_dense_4_bias_read_readvariableop7savev2_custom_critic_dense_5_kernel_read_readvariableop5savev2_custom_critic_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
::	:: 2(
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
:	: 

_output_shapes
::

_output_shapes
: 
ą
Ť
C__inference_dense_3_layer_call_and_return_conditional_losses_695422

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
Ţ
}
(__inference_dense_4_layer_call_fn_695569

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallô
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
GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6954492
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
Ţ
}
(__inference_dense_3_layer_call_fn_695549

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallô
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
GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6954222
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
Ď
Ť
C__inference_dense_5_layer_call_and_return_conditional_losses_695579

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ě
Ŕ
.__inference_custom_critic_layer_call_fn_695510
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCallŽ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_custom_critic_layer_call_and_return_conditional_losses_6954922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

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
Ü
}
(__inference_dense_5_layer_call_fn_695588

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_6954752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ş
ś
$__inference_signature_wrapper_695529
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_6954072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

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
ą
Ť
C__inference_dense_4_layer_call_and_return_conditional_losses_695560

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
 
_user_specified_nameinputs"¸L
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
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:ĎO
Ü
d1
d2
v
trainable_variables
	variables
regularization_losses
	keras_api

signatures
/_default_save_signature
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_modelď{"class_name": "custom_critic", "name": "custom_critic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "custom_critic"}}
ń

	kernel

bias
regularization_losses
	variables
trainable_variables
	keras_api
2__call__
*3&call_and_return_all_conditional_losses"Ě
_tf_keras_layer˛{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 197}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 197]}}
ň

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
4__call__
*5&call_and_return_all_conditional_losses"Í
_tf_keras_layerł{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1024]}}
đ

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
6__call__
*7&call_and_return_all_conditional_losses"Ë
_tf_keras_layerą{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
J
	0

1
2
3
4
5"
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
 "
trackable_list_wrapper
Ę
metrics
layer_metrics
trainable_variables

layers
	variables
regularization_losses
non_trainable_variables
layer_regularization_losses
0__call__
/_default_save_signature
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
,
8serving_default"
signature_map
0:.
Ĺ2custom_critic/dense_3/kernel
):'2custom_critic/dense_3/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
­
 metrics
regularization_losses
!layer_metrics

"layers
	variables
trainable_variables
#non_trainable_variables
$layer_regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
0:.
2custom_critic/dense_4/kernel
):'2custom_critic/dense_4/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
%metrics
regularization_losses
&layer_metrics

'layers
	variables
trainable_variables
(non_trainable_variables
)layer_regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
/:-	2custom_critic/dense_5/kernel
(:&2custom_critic/dense_5/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
*metrics
regularization_losses
+layer_metrics

,layers
	variables
trainable_variables
-non_trainable_variables
.layer_regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
ŕ2Ý
!__inference__wrapped_model_695407ˇ
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
2ţ
.__inference_custom_critic_layer_call_fn_695510Ë
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
2
I__inference_custom_critic_layer_call_and_return_conditional_losses_695492Ë
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
Ň2Ď
(__inference_dense_3_layer_call_fn_695549˘
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
í2ę
C__inference_dense_3_layer_call_and_return_conditional_losses_695540˘
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
Ň2Ď
(__inference_dense_4_layer_call_fn_695569˘
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
í2ę
C__inference_dense_4_layer_call_and_return_conditional_losses_695560˘
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
Ň2Ď
(__inference_dense_5_layer_call_fn_695588˘
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
í2ę
C__inference_dense_5_layer_call_and_return_conditional_losses_695579˘
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
3B1
$__inference_signature_wrapper_695529input_1
!__inference__wrapped_model_695407p	
1˘.
'˘$
"
input_1˙˙˙˙˙˙˙˙˙Ĺ
Ş "3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙Ż
I__inference_custom_critic_layer_call_and_return_conditional_losses_695492b	
1˘.
'˘$
"
input_1˙˙˙˙˙˙˙˙˙Ĺ
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
.__inference_custom_critic_layer_call_fn_695510U	
1˘.
'˘$
"
input_1˙˙˙˙˙˙˙˙˙Ĺ
Ş "˙˙˙˙˙˙˙˙˙Ľ
C__inference_dense_3_layer_call_and_return_conditional_losses_695540^	
0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙Ĺ
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 }
(__inference_dense_3_layer_call_fn_695549Q	
0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙Ĺ
Ş "˙˙˙˙˙˙˙˙˙Ľ
C__inference_dense_4_layer_call_and_return_conditional_losses_695560^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 }
(__inference_dense_4_layer_call_fn_695569Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙¤
C__inference_dense_5_layer_call_and_return_conditional_losses_695579]0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 |
(__inference_dense_5_layer_call_fn_695588P0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ł
$__inference_signature_wrapper_695529{	
<˘9
˘ 
2Ş/
-
input_1"
input_1˙˙˙˙˙˙˙˙˙Ĺ"3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙