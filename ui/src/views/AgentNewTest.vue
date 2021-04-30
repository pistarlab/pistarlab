<template lang="html">
<div>
    <b-modal id="modal-paramedit" footerClass="p-2 border-top-0" title="Edit" size="lg" no-footer>
        <ParamEditor :params="params" :currentParamValues="paramValues" @click="saveParams($event)" ref="peditor"></ParamEditor>
    </b-modal>
    <div class="mt-4">
        <b-button @click="load()" v-b-modal.modal-paramedit>Open</b-button>
    </div>
    <div class="mt-4">
       <pre>
           {{paramValues}}
           </pre>
    </div>
</div>
</template>

<script>
import axios from "axios";
import {
    appConfig
} from "../app.config";
import {
    timedelta,
    timepretty
} from "../funcs";
import gql from "graphql-tag";
import ParamEditor from "../components/ParamEditor.vue";

export default {
    name: "NewAgent",
    components: {
        ParamEditor
    },
    apollo: {

    },
    data() {
        return {
            agentSpec: {},
            options: {},
            component: "b-button",
            params: {
                "trainer.num_workers": {
                    "displayed_name": "Number of Workers",
                    "description": "Number of rollout worker actors to create for parallel sampling. Setting this to 0 will force rollouts to be done in the trainer actor.",
                    "path": "trainer.num_workers",
                    "default": 0,
                    "data_type": "int",
                    "type_info": {

                    }
                },
                "trainer.num_of_envs_per_worker": {
                    "displayed_name": "Number of Environments per Worker",
                    "description": "When num_workers > 0, the driver (local_worker; worker-idx=0) does not need an environment. This is because it doesn't have to sample (done by remote_workers; worker_indices > 0) nor evaluate (done by evaluation workers; see below).",
                    "path": "trainer.num_envs_per_worker",
                    "default": 1,
                    "data_type": "int",
                    "type_info": {

                    }
                },
                // "trainer.create_env_on_driver": {
                //     "displayed_name": "Create Environment on Driver",
                //     "description": "Only applies if if num_of_workers>0",
                //     "path": "trainer.create_env_on_driver",
                //     "default": false,
                //     "data_type": "boolean",
                //     "type_info": {

                //     }
                // },
                "trainer.gamma": {
                    "displayed_name": "Gamma",
                    "description": "",
                    "path": "trainer.gamma",
                    "default": 0.99,
                    "data_type": "float",
                    "type_info": {
                        "min": 0.1,
                        "max": 1.0
                    }
                },
                "trainer.lr": {
                    "displayed_name": "Learning Rate",
                    "description": "",
                    "path": "trainer.lr",
                    "default": 0.0001,
                    "data_type": "float",
                    "type_info": {
                        "min": 0.1,
                        "max": 1.0,
                        'use_range': true
                    }
                },
                "trainer.rollout_fragment_length": {
                    "displayed_name": "Rollout Fragment Length",
                    "description": "",
                    "path": "trainer.rollout_fragment_length",
                    "default": 20,
                    "data_type": "int",
                    "type_info": {
                        "min": 5,
                        "max": 100000,
                        'use_range': true
                    }
                },
                "trainer.soft_horizon": {
                    "displayed_name": "Soft Horizon",
                    "description": "",
                    "path": "trainer.soft_horizon",
                    "default": false,
                    "data_type": "boolean",
                    "type_info": {

                    }
                },
                "trainer.model.fcnet_hiddens": {
                    "displayed_name": "Fully Connected Hiddens",
                    "description": "Size and number of hidden layers to be used. These are used if no custom model is specified and the input space is 1D",
                    "path": "trainer.model.fcnet_hiddens",
                    "default": "[256,256]",
                    "data_type": "list",
                    "type_info": {
                        "sub_type": "int",
                        "default": 256
                    }
                },
                "trainer.model.fcnet_activation": {
                    "displayed_name": "Activation function descriptor",
                    "description": "Activation following each fully connected layer",
                    "path": "trainer.model.fcnet_activation",
                    "default": "tanh",
                    "data_type": "category",
                    "type_info": {
                        "options": ['tanh', 'relu', 'swish', 'silo', 'linear', 'None']
                    }
                },
                "trainer.model.use_lstm": {
                    "displayed_name": "Use LSTM",
                    "description": "Include LSTM for state",
                    "path": "trainer.model.use_lstm",
                    "default": false,
                    "data_type": "boolean",
                    "type_info": {

                    }
                }
            },

            paramValues: {}
        };
    },
    computed: {

        //
    },
    props: {
        specId: String
    },
    methods: {

        cancel() {
            //          
        },
        onError() {
            //
        },
        load() {

            // this.$refs.peditor.loadParams(this.paramValues)

        },
        saveParams(event) {
            this.paramValues = event
        },
    },
    watch: {
        //
    },
    created() {
        //
    },
};
</script>

<style >

</style>
