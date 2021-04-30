<template>
<div>
    <b-button target="_blank" variant="primary" size="sm" :to="`/agent/specs`">New Instance</b-button>
    <b-button variant="secondary" class="ml-2" size="sm" @click="refreshAgents()">Refresh</b-button>
    <div class="mt-4"></div>
    <div v-for="(agent,idx) in agents" v-bind:key="idx">
        <b-row>
            <b-col>
                <div>
                    <router-link target="_blank" :to="`/agent/view/${agent.ident}`"> {{ agent.ident }}</router-link>
                </div>
            </b-col>

            <b-col>
                <div>
                    <b-button size="sm" variant="primary" @click="selectAgent(agent)">Select</b-button>

                </div>
            </b-col>
        </b-row>
        <div class="mt-1"></div>
        <b-row>
            <b-col class="">

                <div>
                    <span class="data_label mt-1">Spec Id: </span>
                    <span>{{agent.specId}}</span>
                </div>
                <div>
                    <span class="data_label mt-1">created: </span>
                    <span>{{agent.created}}</span>
                </div>

            </b-col>
               <b-col class="">
                   </b-col>

        </b-row>

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

export default {
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
            selectedExistingAgent: null

        };
    },
    mounted() {
        //
    },
    methods: {
        selectAgent(agent) {
            this.$emit('click', agent)

        },

        getImageId(uid) {
            let id = parseInt(uid.split("-")[1]);
            return id % 19;
        },
        refreshAgents() {
            this.$apollo.queries.allAgents.refetch();

        },

    },
    computed: {

        agents() {
            if (this.allAgents.length == 0) return [];
            else {
                return this.allAgents.edges.map((v) => v.node);
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
