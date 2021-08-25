<template lang="html">
<div v-if="agentSpec">
    <b-modal id="selectsnapshot" size="xl" scrollable :hide-header="true" :hide-footer="false">
        <div class="text-center mt-4">
            <div class="mt-4"></div>
            <b-alert v-if="errorMessage" show variant="danger">{{ errorMessage }}:
                <pre style="background-color:inherit">{{ traceback }}</pre>
            </b-alert>

            <span v-if="snapshotId" class="h4 mt-4">
                <i class="fa fa-camera"></i> Selected Snapshot: <b>{{snapshotId}}</b>
            </span>
            <span v-else class="h4 mt-4">
                No snapshot selected. A new agent will be created using the default parameters.
            </span>
        </div>

        <div class="mt-4 ml-4 mb-4">
            <b-form-radio v-model="snapshotId" :value="null" name="xxx">None</b-form-radio>
        </div>

        <b-tabs>
            <b-tab title="Local Snapshots" active>
                <div class="mt-4 overflow-auto">
                    <SnapshotSelector :specId="agentSpec.ident" v-model="snapshotId"></SnapshotSelector>
                </div>
            </b-tab>
            <b-tab title="Online Snapshots">
                <div class="mt-4 overflow-auto">

                    <SnapshotSelector :specId="agentSpec.ident" v-model="snapshotId" :online="true"></SnapshotSelector>
                </div>

            </b-tab>
        </b-tabs>
    </b-modal>

    <h3>New Agent Instance</h3>
    <div class="mt-4"></div>
    <b-container fluid v-if="!$apollo.queries.agentSpec.loading">
        <b-row>
            <b-col cols="4" class="text-center">
                <b-card-img :src="`/img/agent_spec_icons/agent_${getImageId(agentSpec.ident)}.png`" alt="Image" style="max-width:200px;"></b-card-img>

            </b-col>

            <b-col cols=2 class="text-right">

                <div>Agent Spec:</div>

                <div>Extension ID:</div>

                <div>Version:</div>
            </b-col>
            <b-col>
                <div>
                    <span>
                        <router-link :to="`/agent_spec/${agentSpec.ident}`">{{agentSpec.displayedName}}</router-link>
                    </span>
                </div>
                <div>
                    <span>{{ agentSpec.extensionId }}</span>
                </div>
                <div>
                    <span>{{ agentSpec.version }}</span>
                </div>
            </b-col>
        </b-row>
        <div class="mt-2"></div>
        <b-row>
            <b-col cols="4" class="text-right">
            </b-col>
            <b-col>
                <div>
                    <span>{{ agentSpec.description }}</span>
                </div>
            </b-col>
        </b-row>
        <div class="mt-4"></div>
        <b-row>
            <b-col>
                <div>
                    <b-tabs content-class="mt-3" justified>
                        <b-tab title="Create" active class="text-center">
                            <p>Create an agent with default parameters or select one from an existing snapshot.</p>
                            <b-row class="mt-4 ml-4 mb-4 ">
                                <b-col>

                                    <b-button variant="info" size="sm" v-b-modal:selectsnapshot class="mr-4">Select an Agent Snapshot</b-button>
                                </b-col>
                                <b-col>
                                    <b-form-radio class="mt-2" v-model="snapshotId" :value="null" name="xxx">New Agent with default parameters.</b-form-radio>
                                    <!-- <span v-if="snapshotId">
                                         {{snapshotId}}
                                </span>
                                <span v-else>
                                    None selected
                                </span> -->

                                </b-col>
                            </b-row>

                            <div class="text-center mt-4">
                                <div class="mt-4"></div>
                                <b-alert v-if="errorMessage" show variant="danger">{{ errorMessage }}:
                                    <pre style="background-color:inherit">{{ traceback }}</pre>
                                </b-alert>

                                <span v-if="snapshotId" class="h4 mt-4">
                                    <i class="fa fa-camera"></i> Selected Snapshot<br /> <b>{{snapshotId}}</b>
                                </span>
                                <span v-else class="h4 mt-4">
                                    No snapshot selected. A new agent will be created using the default parameters.
                                </span>
                            </div>
                            <!-- <b-tabs>
                                <b-tab title="Local Snapshots" active>
                                    <div class="mt-4 overflow-auto">
                                        <SnapshotSelector :specId="agentSpec.ident" v-model="snapshotId"></SnapshotSelector>
                                    </div>
                                </b-tab>
                                <b-tab title="Online Snapshots">
                                    <div class="mt-4 overflow-auto">

                                        <SnapshotSelector :specId="agentSpec.ident" v-model="snapshotId" :online="true"></SnapshotSelector>
                                    </div>

                                </b-tab>
                            </b-tabs> -->

                        </b-tab>
                        <b-tab title="Create with Agent Builder">
                            <div v-if="params">
                                <div class="ml-3">
                                    <ParamEditor :params="params" :values="initParamValues" @update="saveParams($event)" ref="peditor"></ParamEditor>
                                </div>
                            </div>
                            <div v-else>
                                Agent Builder not supported by this Agent Spec
                            </div>
                            <div class="mt-4"></div>
                            <b-alert v-if="errorMessage" show variant="danger">{{ errorMessage }}:
                                <pre style="background-color:inherit">{{ traceback }}</pre>
                            </b-alert>

                        </b-tab>

                        <b-tab title="Created from raw config">

                            <editor v-if="config" v-model="config" @init="editorInit" lang="json" width="100%" theme="chrome" height="600"></editor>
                            <div class="mt-4"></div>
                            <div class="mt-4"></div>
                            <b-alert v-if="errorMessage" show variant="danger">{{ errorMessage }}:
                                <pre style="background-color:inherit">{{ traceback }}</pre>
                            </b-alert>

                        </b-tab>
                    </b-tabs>

                </div>

            </b-col>
        </b-row>

    </b-container>
    <div class="mt-4"></div>
    <b-button-toolbar>
        <b-button size="sm" class="ml-auto" v-if="!submitting" variant="primary" v-on:click="submit">Create Instance</b-button>
        <b-button size="sm" class="ml-auto" v-else variant="primary" disabled>
            <b-spinner small type="grow"></b-spinner>Processing...
        </b-button>
    </b-button-toolbar>
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
      displayedName
      description
      extensionId
      version
      config
      params
    }
  }
`;
import ParamEditor from "./ParamEditor.vue";

import SnapshotSelector from "./SnapshotSelector.vue";

export default {
    components: {
        editor: require('vue2-ace-editor'),
        SnapshotSelector,
        ParamEditor
    },
    apollo: {
        agentSpec: {
            query: GET_AGENT_SPEC,
            variables() {
                return {
                    ident: this.specId,
                };
            },
        },
    },
    data() {
        return {
            agentSpec: {},
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
        specId: String
    },
    computed: {
        params() {
            if (!this.agentSpec) return null
            return JSON.parse(this.agentSpec.params)

        },
        initParamValues() {
            if (!this.agentSpec.config) return null
            return JSON.parse(this.agentSpec.config)
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
            const outgoingData = {
                config: JSON.parse(this.config),
                specId: this.specId,
                snapshotId: this.snapshotId

            }
            axios
                .post(`${appConfig.API_URL}/api/new_agent_submit`, outgoingData)
                .then((response) => {
                    const data = response.data["item"];
                    if ("uid" in data) {
                        this.$emit("agentCreated", data.uid)

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
        },
        saveParams(event) {
            this.paramValues = event
            this.config = JSON.stringify({
                ...JSON.parse(this.config),
                ...this.paramValues
            }, null, 2);
        }
    },
    watch: {
        agentSpec: function (val) {
            this.agentSpec = val;
            if (this.agentSpec.config) {
                this.config = JSON.stringify(JSON.parse(this.agentSpec.config), null, 2)
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
