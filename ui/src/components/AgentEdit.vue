<template lang="html">
<div v-if="agentSpec">
    <b-container fluid v-if="!$apollo.queries.agent.loading">
        <b-row>
            <b-col>

                <div>
                    <b-tabs content-class="mt-3" justified>
                        <b-tab title="Edit with Agent Builder">
                            <div v-if="params">
                                <div class="ml-3">
                                    <ParamEditor :params="params" :values="initParamValues" @update="paramValues = $event">
                                    </ParamEditor>
                                </div>
                            </div>
                            <div v-else>
                                Agent Builder not supported by this Agent Spec
                            </div>

                        </b-tab>
                        <b-tab title="Edit raw config">

                            <editor v-if="config" v-model="config" @init="editorInit" lang="json" width="100%" theme="chrome" height="600"></editor>
                            <div class="mt-4"></div>
                        </b-tab>
                    </b-tabs>

                </div>

            </b-col>
        </b-row>
        <div class="mt-4"></div>
    </b-container>
    <b-button-toolbar>
        <b-button size="sm" class="ml-auto" v-if="!submitting" variant="primary" v-on:click="submit">Save</b-button>
        <b-button size="sm" class="ml-auto" v-else variant="primary" disabled>
            <b-spinner small type="grow"></b-spinner>Processing...
        </b-button>
    </b-button-toolbar>
    <div class="mt-4"></div>

    <b-alert v-if="errorMessage" show variant="danger">{{ errorMessage }}:
        <pre style="background-color:inherit">{{ traceback }}</pre>
    </b-alert>

</div>
</template>

<script>
//USING https://github.com/chairuosen/vue2-ace-editor
import axios from "axios";
import {
    appConfig
} from "../app.config";
import {
    timedelta,
    timepretty
} from "../funcs";
import gql from "graphql-tag";

const GET_AGENT_SPEC = gql `
  query GetAgentSpec($ident: String!) {
    agentSpec(ident: $ident) {
      id
      ident
      description
      extensionId
      version
      config
      params
    }
  }
`;

const GET_AGENT = gql `
  query GetAgent($ident: String!) {
    agent(ident: $ident) {
      id
      ident
      config
      spec{
        id
        ident
        description
        extensionId
        version
        config
        params
      }
    }
  }
`;

import ParamEditor from "./ParamEditor.vue";

export default {
    name: "EditAgent",
    components: {
        editor: require('vue2-ace-editor'),
        ParamEditor
    },
    apollo: {
        agent: {
            query: GET_AGENT,
            variables() {
                return {
                    ident: this.uid,
                };
            },
        },
    },
    data() {
        return {
            agent: {},
            options: {},
            config: "",
            code: '',
            paramValues: {},
            submitting: false,
            snapshotId: null,
            snapshot_version: "0",
            snapshot_description: "",
            errorMessage: null
        };
    },
    props: {
        uid: String
    },
    computed: {
        agentSpec() {
            if (!this.agent.spec) return null
            else {
                return this.agent.spec
            }

        },
        params() {
            if (!this.agentSpec) return null
            return JSON.parse(this.agentSpec.params)
            
        },
        initParamValues() {
            if (!this.config || this.config == '') return {}

            return JSON.parse(this.config)
        }

    },

    methods: {
        editorInit: function () {
            require('brace/ext/language_tools') //language extension prerequsite...
            require('brace/mode/html')
            require('brace/mode/json') //language
            require('brace/mode/less')
            require('brace/theme/chrome')
            require('brace/theme/twilight')
            require('brace/snippets/javascript') //snippet
        },
        cancel() {
            //          
        },
        onError() {
            //
        },
        selectSnapshot(snapshotId) {
            this.$bvModal.hide("modal-selectsnapshot");
            this.snapshotId = snapshotId;
        },
        submit() {
            this.submitting = true
            this.config = JSON.stringify({
                ...JSON.parse(this.config),
                ...this.paramValues
            }, null, 2);

            this.$apollo.mutate({
                // Query
                mutation: gql `mutation agentConfigMutation($id:String!,$config:String!) 
                {
                    agentSetConfig(id:$id, config:$config){
                        success
                        }
                }`,
                // Parameters
                variables: {
                    id: this.agent.id,
                    config: this.config
                },

            }).then((data) => {
                console.log("Submitted")
                this.submitting = false
                this.$emit("updated")

            }).catch((error) => {
                // Error
                this.submitting = false
                console.error(error)
                // We restore the initial user input
            })

        }
    },
    watch: {
        agent: function (val) {
            this.agent = val;
            if (this.agent.config) {
                this.config = JSON.stringify(JSON.parse(this.agent.config), null, 2)
                //
            }
        },
    },
    created() {

        //
    },
};
</script>

<style scoped>
.ace_editor {
    font-size: 16px;
}
</style>
