<template>
<div>
    <h3>Agent Specs</h3>
    <b-container fluid>
        <div v-if="$apollo.queries.agentSpecs.loading">Loading..</div>
        <div v-else-if="agentSpecs && agentSpecs.length > 0">
            <div v-for="item in agentSpecs" :key="item.id">
                <b-row class="pt-4" v-if="!item.disabled">

                    <b-col class="">
                        <h3>
                            <router-link :to="`/agent_spec/${item.ident}`">{{item.displayedName}}</router-link>
                        </h3>
                        <div class="">
                            <b-alert size="sm" v-if="item.disabled" show variant="warning">disabled</b-alert>
                        </div>
                        <!-- <div class="small data_label">Class/Module</div>
                      <span>{{ item.classHame }}/{{ item.module }}</span>-->
                        <div>
                            <span class="data_label mt-1">Plugin/Version: </span>
                            <span v-if="item.pluginId =='WORKSPACE'">
                                <b-badge variant="warning"><i class="fa fa-code"></i> in your workspace</b-badge>
                            </span>
                            <span v-else>
                                {{ item.pluginId }}/{{ item.version }}
                            </span>
                        </div>
                        <div>
                            <div class="data_label">Description: </div>
                            <span>
                                {{ item.description }}
                            </span>
                        </div>
                    </b-col>
                    <b-col cols="4">
                        <b-button class="float-right" size="sm" :disabled=item.disabled @click="select  (item.ident)" variant="outline-primary">
                            Select
                        </b-button>
                    </b-col>
                </b-row>
                <hr />
            </div>
        </div>
        <div v-else>
            <b-row>
                <b-col>{{ message }}</b-col>
            </b-row>
        </div>
    </b-container>
</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import {
    appConfig
} from "../app.config";
import gql from "graphql-tag";

export default {
    components: {
        // SessionList
    },
    apollo: {
        // Simple query that will update the 'hello' vue property
        agentSpecs: gql `
      query {
        agentSpecs: agentSpecs {
          id
          ident
          displayedName
          disabled
          pluginId
          description
          version
        }
      }
    `,
    },
    data() {
        return {
            selectedSpecId: null,
            agentSpecs: [],
            searchQuery: "",
            error: "",
            message: "No Agent Specs Available",
        };
    },

    computed: {
        // items() {
        //   return this.qlitems
        // }
    },
    methods: {
        select(specId) {
            this.$emit("specSelected", specId)
        }

    },
    // Fetches posts when the component is created.
    created() {
        //
    },
};
</script>
