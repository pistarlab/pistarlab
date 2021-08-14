<template lang="html">
<div>
    <h2><i class="fas fa-stream"></i> New Task</h2>
    <div class="mt-4"></div>

    <b-container fluid>
        <b-row>
            <b-col>
                <div class="pt-2">
                    <div class="data_label">Task Spec</div>
                    <span>{{ spec.ident }}</span>
                </div>
                <div class="pt-2">
                    <div class="data_label">Extension ID</div>
                    <span>{{ spec.extensionId }}</span>
                </div>
                <div class="pt-2">
                    <div class="data_label">Version</div>
                    <span>{{ spec.version }}</span>
                </div>
                <div class="pt-2">
                    <div class="data_label">Description</div>
                    <span>{{ spec.description }}</span>
                </div>
            </b-col>
        </b-row>
        <div class="mt-4"></div>

        <b-row>
            <b-col>
                <div class="data_label">Config</div>
                <div>
                    <editor v-model="config" @init="editorInit" lang="json" theme="twilight" width="100%" height="600"></editor>
                </div>
            </b-col>
        </b-row>
        <div class="mt-4"></div>

        <b-row>
            <b-col>
                <div>
                    <b-button size="sm" class="mr-4" v-if="!submitting" variant="primary" v-on:click="submit">Create Instance</b-button>
                    <b-button size="sm" class="mr-4" v-else variant="primary" disabled>
                        <b-spinner small type="grow"></b-spinner>Processing...
                    </b-button>
                    <b-button v-if="!submitting" variant="secondary" size="sm" :to="`/task/specs/`">Cancel</b-button>
                </div>
            </b-col>
        </b-row>
    </b-container>
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

const GET_TASK_SPEC = gql `
  query GetTaskSpec($ident: String!) {
    spec:taskSpec(ident: $ident) {
      id
      ident
      description
      extensionId
      version
      config
    }
  }
`;

export default {
    name: "NewTask",
    components: {
        editor: require('vue2-ace-editor'),
    },
    apollo: {
        spec: {
            query: GET_TASK_SPEC,
            variables() {
                return {
                    ident: this.specId,
                };
            },
        },
    },
    data() {
        return {
            spec: {},
            options: {},
            config: "",
            code: '',
            submitting: false,
        };
    },
    props: {
        specId: String
    },
    methods: {
        editorInit: function () {
            require('brace/ext/language_tools') //language extension prerequsite...
            require('brace/mode/html')
            require('brace/mode/json') //language
            require('brace/mode/less')
            require('brace/theme/twilight')
            require('brace/snippets/javascript') //snippet
        },
        cancel() {
            //          
        },
        onError() {
            //
        },
        submit() {
            const outgoingData={
                config:JSON.parse(this.config),
                specId:this.specId

            }
             axios
            .post(`${appConfig.API_URL}/api/new_task_submit`, outgoingData)
            .then((response) => {
                const data = response.data["item"];
                if ("uid" in data) {
                    this.$router.push({
                        path: `/task/view/${data.uid}`,
                    });
                } else {
                    console.log("ERROR in response " + JSON.stringify(data));
                    this.errorMessage = JSON.stringify(data["error"]);
                }
                this.traceback = data["traceback"];

                this.submitting = false;
            })
            .catch(function (error) {
                this.errorMessage = error;
                this.submitting = false;
            });
        }
    },
    watch: {
        spec: function (val) {
            this.spec = val;
            if (this.spec.config) {
                this.config = JSON.stringify(JSON.parse(this.spec.config),null,2)
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
