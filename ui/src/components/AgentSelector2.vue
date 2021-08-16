<template>
<div>
    <b-modal id="agentnew" size="xl" :hide-header="true" :hide-footer="true">
        <AgentNew :specId="selectedSpecId" @agentCreated="agentCreated($event)"></AgentNew>
    </b-modal>
    <b-modal id="agentspecs" size="xl" :hide-header="true" :hide-footer="true">
        <AgentSpecs @specSelected="newAgentModal($event)"></AgentSpecs>
    </b-modal>
    <b-button-toolbar>
        <b-button target="_blank" variant="primary" size="sm" v-b-modal:agentspecs>New Agent Instance</b-button>
        <b-button variant="secondary" class="ml-auto" size="sm" @click="refreshAgents()">Refresh List </b-button>
        <b-form-input v-model="searchtext" placeholder="Search" style="width:250px;" class='ml-auto'></b-form-input>

    </b-button-toolbar>
    <div class="mt-4"></div>
    <div v-for="(agent,idx) in agents" v-bind:key="idx">
        <AgentRowSelect :agent="agent" @click="selectAgent(agent)" />

        <hr />
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

const existingAgentfields = [{
        key: "label",
        label: "UID",
    },
    {
        key: "specId",
        label: "Spec Id",
        sortable: true,
    },

    {
        key: "created",
        label: "Creation Time",
        sortable: true,
    },
    {
        key: "link",
        label: "",
    },
];

const GET_AGENT_SPECS = gql `
  query {
    agentSpecs {
      id
      ident
      created
      displayedName
      config
      params
    }
  }
`;

const GET_AGENTS = gql `
  {
    allAgents {
      edges {
        node {
          id
          ident
          displayedName
          created
          specId

          config
          status
        }
      }
    }
  }
`;
import AgentRowSelect from "../components/AgentRowSelect.vue";
import AgentSpecs from "../components/AgentSpecs.vue";
import AgentNew from "../components/AgentNew.vue";
export default {
    components: {
        AgentRowSelect,
        AgentSpecs,
        AgentNew

    },
    props: {
        //
    },
    apollo: {
        agentSpecs: GET_AGENT_SPECS,
        allAgents: GET_AGENTS,

    },
    data() {
        return {
            allAgents: [],
            selectedExistingAgent: null,
            selectedSpecId: null,
            searchtext: ""

        };
    },
    mounted() {
        //
    },
    methods: {
        agentCreated(agentId) {
            this.$bvModal.hide("agentnew")
            this.refreshAgents()
        },
        newAgentModal(specId) {
            this.selectedSpecId = specId
            this.$bvModal.hide("agentspecs")
            this.$bvModal.show("agentnew")
        },
        selectAgent(agent) {
            this.$emit('click', agent.ident)

        },

       
        refreshAgents() {
            this.$apollo.queries.allAgents.refetch();

        },

    },
    computed: {

        agents() {
            if (this.allAgents.length == 0) return [];
            else {
                let agents = this.allAgents.edges.map((v) => v.node);
                if (this.searchtext != "") {
                    return agents.filter((v) => {
                        var vals = [v.displayedName, v.name, v.specId]
                        var keep = false
                        for (let st of vals) {
                            if (st != null && st.toLowerCase().includes(this.searchtext.toLowerCase())) {
                                keep = true
                                break;
                            }

                        }
                        return keep

                    }).sort((a,b) =>  b.created - a.created)
                } else {
                    return agents.sort((a,b) =>  b.created - a.created)

                }

            }
        }
        //

    },
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style scoped lang="scss">

</style>
