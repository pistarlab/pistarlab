<template lang="html">
<div>
    <b-modal id="agentnew" size="xl" :hide-header="true" :hide-footer="true">
        <AgentNew :specId="agentSpec.ident" @agentCreated="agentCreated($event)"></AgentNew>
    </b-modal>
    <h1><i class="fa fa-robot"></i> Agent Spec: <span v-if="agentSpec">{{agentSpec.displayedName}}</span></h1>
    <div class="mt-4"></div>
    <b-button size="sm" :disabled=agentSpec.disabled v-b-modal:agentnew variant="primary">New Instance</b-button>
    <div class="mt-2"></div>

    <b-container fluid>

        <b-row>
            <b-col cols=3 class="text-center">
                <b-card-img class="mt-4" :src="`/img/agent_spec_icons/agent_${getImageId(agentSpec.ident)}.png`" alt="Image" style="max-width:300px;"></b-card-img>
                <div class="m-4">
                    <b-link target="_blank" :href="`http://github.com/pistarlab/pistarlab/wiki/${agentSpec.ident}`">Wiki</b-link>
                </div>
            </b-col>
            <b-col>
                <div class="pt-2">
                    <div class="data_label">Name</div>
                    <span>{{ agentSpec.displayedName }}</span>
                </div>
                <div class="pt-2">
                    <div class="data_label">Spec Id</div>
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
                    <span style=" white-space: pre-wrap;">{{ agentSpec.description }}</span>
                </div>
                <div class="pt-2">
                    <div class="data_label">Usage</div>
                    <span style=" white-space: pre-wrap;">{{ agentSpec.usage }}</span>
                </div>

            </b-col>
        </b-row>
        <div class="mt-4"></div>
    </b-container>

    <hr />

    <b-button v-b-toggle.collapse-details variant="secondary">Configuration Details</b-button>

    <b-collapse id="collapse-details" class="mt-2">
        <b-card>
            <b-container fluid>
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

    </b-collapse>


</div>
</template>

<script>

import gql from "graphql-tag";
import AgentNew from "../components/AgentNew.vue";

const GET_AGENT_SPEC = gql `
  query GetAgentSpec($ident: String!) {
    agentSpec(ident: $ident) {
      id
      ident
      displayedName
      description
      collection
      usage
      extensionId
      version
      config
      params
      disabled
    }
  }
`;

export default {
    name: "AgentSpecView",
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
