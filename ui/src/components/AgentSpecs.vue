<template>
<div>
    <h3>Agent Specs</h3>
    <b-form-input v-model="searchtext" placeholder="Search" style="width:250px;"></b-form-input>

    <b-container fluid>
        <div v-if="$apollo.queries.agentSpecs.loading">Loading..</div>
        <div v-else-if="agentSpecs && agentSpecs.length > 0">
            <div v-for="item in specs" :key="item.id">
                <b-row class="pt-4" v-if="!item.disabled">
                    <b-col cols="2" class="text-center">
                        <b-card-img :src="`/img/agent_spec_icons/agent_${getImageId(item.ident)}.png`" alt="Image" style="max-width:60px;"></b-card-img>

                    </b-col>

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
                            <span class="data_label mt-1">Spec Id: </span>
                            <span>
                                {{ item.ident }}
                            </span>
                        </div>
                        <div>

                            <span v-if="item.extensionId =='WORKSPACE'">
                                <b-badge variant="warning"><i class="fa fa-code"></i> in your workspace</b-badge>
                            </span>
                            <span class="data_label mt-1" v-else> Extension/Version </span> {{ item.extensionId }}/{{ item.version }}

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
          extensionId
          version
        }
      }
    `,
    },
    data() {
        return {
            selectedSpecId: null,
            agentSpecs: [],
            searchtext: "",
            error: "",
            message: "No Agent Specs Available",
        };
    },

    computed: {
        specs() {
            if (this.agentSpecs == null || this.agentSpecs.length == 0) return [];
            else {
                if (this.searchtext != "") {
                    return this.agentSpecs.filter((v) => {
                        var vals = [v.displayedName, v.ident, v.extensionId]
                        var keep = false
                        for (let st of vals) {
                            if (st != null && st.toLowerCase().includes(this.searchtext.toLowerCase())) {
                                keep = true
                                break;
                            }

                        }
                        return keep

                    })
                } else {
                    return this.agentSpecs
                }

            }
        }
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
