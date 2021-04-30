<template>
<div>
    <div v-for="(agent,idx) in allAgents.edges" v-bind:key="idx">
           <b-row>
                <b-col>
                    <div>
                        <router-link :to="`/agent/view/${agent.ident}`"> {{ agent.ident }}</router-link>
                    </div>
                </b-col>

                <b-col>
                    <div>
                        <b-button size="sm" variant="primary" @click="select(agent.ident)">Select</b-button>

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
                        <span>{{spec.created}}</span>
                    </div>

                </b-col>
                <b-col class="">
                    <div>
                        <span class="data_label mt-1">Type: </span>
                        <span>{{spec.envType}}</span>
                    </div>
                    <div>
                        <span class="data_label mt-1">Description: </span>
                        <span>{{spec.description}}</span>
                    </div>
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
          ident
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
            selectedExistingAgent:null

        };
    },
    mounted() {
        //
    },
    methods: {
        selectAgent(agentId){
            this.$emit('click',agentId)

        },

        getImageId(uid) {
            let id = parseInt(uid.split("-")[1]);
            return id % 19;
        },

    },
    computed: {
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
