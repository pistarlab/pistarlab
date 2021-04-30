<template>
<div>
    <h1>Agents Instances</h1>

    <div class="mt-4"></div>
    <div v-if="$apollo.queries.allAgents.loading">Loading..</div>

    <div v-else>
        <b-button :disabled="selected.length <= 1" v-on:click="runCompare" variant="info" size="sm">
            <span v-if="selected.length > 1">Compare: {{selected.join(", ")}}</span>
            <span v-else-if="selected.length == 1">Compare: selected one more ({{selected.join(", ")}}) </span>

            <span v-else>Select 2-5 to compare</span> 

        </b-button>
        <div class="mt-3">
    </div>
        <b-alert show variant="warning">
            WARNING: Comparing agents is under development and currently broken.
        </b-alert>
        <b-card>

            <b-form-checkbox-group v-model="selected">

                <b-table show-empty empty-text="No Instances Found" hover table-busy :items="items" :fields="fields" :dark="false" :small="false" :bordered="false" :outlined="false" :borderless="false">
                    <template v-slot:cell(selector)="data">
                        <b-form-checkbox :value="data.item.ident"></b-form-checkbox>
                    </template>
                    <template v-slot:cell(link)="data">
                        <!-- `data.value` is the value after formatted by the Formatter -->
                        <router-link :to="`/agent/view/${ data.item.ident}`">{{ data.item.ident }}</router-link>
                    </template>
                </b-table>
                <p>{{error}}</p>

            </b-form-checkbox-group>

        </b-card>
    </div>
</div>
</template>
// #0d1117
<script>
// @ is an alias to /src
import axios from "axios";
import {
    appConfig
} from "../app.config";

import {
    timedelta,
    timepretty
} from "../funcs";
import gql from "graphql-tag";
//TODO: pageinate
//https://jeffersonheard.github.io/python/graphql/2018/12/08/graphene-python.html
const GET_AGENTS = gql `
{
  allAgents(sort:CREATED_DESC) { #, first:10
        pageInfo {
      startCursor
      endCursor
            hasNextPage
      hasPreviousPage
    }
    edges {
      node {
        ident
        created
        specId
      }
    }
  }
}
`;
const fields = [{
        key: "selector",
        label: "",
    },
    {
        key: "link",
        label: "Label",
        sortable: true,
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
    }
];

export default {
    name: "Agents",
    components: {
        // SessionList
    },
    apollo: {
        // Simple query that will update the 'hello' vue property
        allAgents: GET_AGENTS,
    },
    data() {
        return {
            searchQuery: "",
            allAgents: [],
            selected: [],
            error: "",
            taskEntities: [],
            fields: fields,
        };
    },

    computed: {
        items() {
            if (this.allAgents.edges == null)
                return []

            return this.allAgents.edges.map((edge) => edge.node)
        },
    },
    methods: {
        timedelta,
        runCompare() {
            this.$router.push({
                path: `/agents/compare?uids=` + this.selected.join(","),
            });
        },
    },
    // Fetches posts when the component is created.
    created() {
        //
    },
};
</script>
