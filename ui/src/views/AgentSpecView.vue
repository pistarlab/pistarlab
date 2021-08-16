<template lang="html">
<div>
    <b-modal id="agentnew" size="xl" :hide-header="true" :hide-footer="true">
        <AgentNew :specId="agentSpec.ident" @agentCreated="agentCreated($event)"></AgentNew>
    </b-modal>
    <h1><i class="fa fa-robot"></i> Agent Spec</h1>
    <div class="mt-4"></div>
    <b-button size="sm" :disabled=agentSpec.disabled v-b-modal:agentnew variant="primary">New Instance</b-button>
    <div class="mt-2"></div>
    <b-card-img :src="`/img/agent_spec_icons/agent_${getImageId(agentSpec.ident)}.png`" alt="Image"  style="max-width:60px;"></b-card-img>

    <b-card>
        <b-container fluid>
            
            <b-row>
                <b-col>
                    <div class="pt-2">
                        <div class="data_label">Agent Spec</div>
                        <span>{{ agentSpec.ident }}</span>
                    </div>
                    <div class="pt-2">
                        <div class="data_label">Extension ID</div>
                        <span>{{ agentSpec.extensionId }}</span>
                    </div>
                    <div class="pt-2">
                        <div class="data_label">Version</div>
                        <span>{{ agentSpec.version }}</span>
                    </div>
                    <div class="pt-2">
                        <div class="data_label">Description</div>
                        <span>{{ agentSpec.description }}</span>
                    </div>
                </b-col>
            </b-row>
            <div class="mt-4"></div>

            <b-row>
                <b-col>
                    <div class="data_label">Default Config</div>
                    <div v-if="agentSpec && agentSpec.config">
                        <pre>{{JSON.parse(agentSpec.config)}}</pre>
                    </div>
                </b-col>
            </b-row>
            <div class="mt-4"></div>
            <b-row>
                <b-col>
                    <div class="data_label">Params</div>
                    <div v-if="agentSpec && agentSpec.params">
                        <pre>{{JSON.parse(agentSpec.params)}}</pre>
                    </div>
                </b-col>
            </b-row>
        </b-container>
    </b-card>
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
import AgentNew from "../components/AgentNew.vue";

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
      disabled
    }
  }
`;

export default {
    name: "NewAgent",
    components: {
        AgentNew
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
            submitting: false,
        };
    },
    props: {
        specId: String
    },
    methods: {
        agentCreated(agentId) {
            this.$router.push({
                path: `/agent/view/${agentId}`,
            });

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

<style scoped>
.ace_editor {
    font-size: 16px;
}
</style>
