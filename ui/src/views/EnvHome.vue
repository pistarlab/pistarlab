<template>
<div>
    <h1><i class="fa fa-gamepad"></i> Environments</h1>
<!-- 
    <b-breadcrumb>
        <b-breadcrumb-item :to="`/`"><i class="fa fa-home"></i></b-breadcrumb-item>
        <b-breadcrumb-item active><i class="fa fa-gamepad"></i> Environments</b-breadcrumb-item>
    </b-breadcrumb> -->

    <div class="mt-4"></div>

     <b-modal id="modal-addcustom" >
         TODO
     </b-modal>
    <b-modal id="modal-selected" footerClass="p-2 border-top-0" :title="`${selectedEnvironment.ident}`" size="lg" scrollable ok-only>

        <b-container fluid>
            <b-row>
                <b-col>

                                        <div>
                        <span class="data_label  mt-2">Categories: </span>
                        <span class="data_label">{{ selectedEnvironment.categories }}</span>
                    </div>
                    <div>
                        <span class="data_label  mt-2">Version: </span>
                        <span class="data_label">{{ selectedEnvironment.version }}</span>
                    </div>

                    <div>
                        <span class="data_label  mt-2">Plugin: </span>
                        <span class="data_label">{{ selectedEnvironment.pluginId }}: {{ selectedEnvironment.pluginVersion }}</span>
                    </div>

                    <div>
                        <span class="data_label mt-2">Description: </span>
                        <span>{{selectedEnvironment.description}}</span>
                    </div>
                </b-col>
                <b-col>
                    <b-button v-b-modal:modal-addcustom size="sm">Add Custom Spec</b-button>
                </b-col>
            </b-row>
        </b-container>

        <div class="mt-4">
        </div>
        <h3>Environment Specs</h3>
        <b-container fluid>
            <div v-for="spec in selectedEnvironment.specs" :key="spec.ident">
                <b-row>
                    <b-col>
                        <div>
                            <router-link :to="`/env_spec/view/${spec.ident}`"> {{ spec.displayedName }}</router-link>
                        </div>
                    </b-col>

                    <b-col>
                        <div>
                            <b-button variant="primary" :to="`/task/new/agenttask/?envSpecId=${spec.ident}`" size="sm">Assign</b-button>

                        </div>
                    </b-col>
                </b-row>
                <div class="mt-1"></div>
                <b-row>
                    <b-col class="">

                        <div>
                            <span class="data_label mt-1">Version: </span>
                            <span>{{spec.version}}</span>
                        </div>
                                                <div>
                            <span class="data_label mt-1">Tags: </span>
                            <span>{{spec.tags}}</span>
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
        </b-container>

    </b-modal>

    <div class="mt-4"></div>

    <b-container fluid>
        <div v-if="$apollo.queries.environments.loading">Loading..</div>
        <div v-else>
            <div v-if="items.length > 0">
                <b-row>
                    <b-col class="d-flex flex-wrap mb-4">
                        <span v-for="(item, idx) in items" v-bind:key="idx" class="m-2" @click="selectGroup(idx)" v-b-modal.modal-selected style="min-width:150px">
                            <EnvironmentCard :item="item"></EnvironmentCard>
                        </span>
                    </b-col>
                </b-row>
            </div>
            <div v-else>{{ message }}</div>

        </div>
    </b-container>
    <br />
    <div class="mt-4"></div>
</div>
</template>

<script>
// @ is an alias to /src
import gql from "graphql-tag";
import {
    appConfig
} from "../app.config";
import EnvironmentCard from "../components/EnvCardGroup.vue";

export default {
    name: "Env",
    components: {
        EnvironmentCard
    },
    apollo: {
        environments: gql `
      query {
        environments: allEnvironments {
          edges {
            node {
                id
                ident
                description
                displayedName
                pluginId
                categories
                pluginVersion
                version
                specs {
                    id
                    ident
                    description
                    displayedName
                    tags
                    envType
                    entryPoint
                }
            }
          }
        }
      }
    `,
    },
    data() {
        return {
            environments: [],
            searchQuery: "",
            selectedEnvironment: {},
            message: "No Environments found.",
            appConfig
        };
    },
    methods: {
        selectGroup(idx) {
            this.selectedEnvironment = this.environments.edges[idx].node;

        },
    },

    computed: {
        items() {
            if (this.environments.length == 0) return [];
            else {
                return this.environments.edges.map((v) => v.node);
            }
        }
    },
    created() {
        //
    },
};
</script>

<style >
</style>
